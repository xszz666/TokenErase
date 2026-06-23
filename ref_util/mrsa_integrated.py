
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class MultiReferenceSelfAttention():
    def __init__(self,  start_step=0, end_step=50, step_idx=None, layer_idx=None, ref_masks=None, mask_weights=[1.0,1.0,1.0], style_fidelity=1, viz_cfg=None):
        """
        Args:
            start_step   : the step to start transforming self-attention to multi-reference self-attention
            end_step     : the step to end transforming self-attention to multi-reference self-attention
            step_idx     : list of the steps to transform self-attention to multi-reference self-attention
            layer_idx    : list of the layers to transform self-attention to multi-reference self-attention
            ref_masks    : masks of the input reference images
            mask_weights : mask weights for each reference masks
            viz_cfg      : config for visualization
        """
        self.cur_step       =  0
        self.num_att_layers = -1
        self.cur_att_layer  =  0

        self.start_step   = start_step
        self.end_step     = end_step
        self.step_idx     = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.layer_idx    = layer_idx
        
        self.ref_masks    = ref_masks
        self.mask_weights = mask_weights
        
        self.style_fidelity = style_fidelity

        self.viz_cfg = viz_cfg

         # 添加状态控制
        self.enabled = False
        self.use_reference = False      
    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):

        if not self.enabled or not self.use_reference or is_cross:
            # 不使用MRSA，直接返回标准注意力
            out = self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            # 使用MRSA
            out = self.mrsa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        # 更新层计数（只在MRSA启用时）
        if self.enabled:
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                
        return out
    
    def get_ref_mask(self, ref_mask, mask_weight, H, W):
        ref_mask = ref_mask.float() * mask_weight
        ref_mask = F.interpolate(ref_mask, (H, W))
        ref_mask = ref_mask.flatten()
        return ref_mask
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        # 确保 dtype 对齐，避免 Half/Float 混用
        if attn is not None and v is not None and attn.dtype != v.dtype:
            v = v.to(attn.dtype)
        if q is not None and k is not None and q.dtype != k.dtype:
            k = k.to(q.dtype)
            if v is not None and v.dtype != q.dtype:
                v = v.to(q.dtype)

        # 在进入 MRSA 前，检查 dtype
        # print("x/q/k/v dtype:", x.dtype)
        # 在 MRSA 内 attn_batch 前后打印：
        # if attn is not None :
            # print("attn dtype:", attn.dtype)
        # if v is not None:
            # print("v dtype:",q.dtype,k.dtype, v.dtype)

        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) 
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        
        if kwargs.get("attn_batch_type") == 'mrsa':
            sim_own, sim_refs = sim[..., :H*W], sim[..., H*W:]
            sim_or = [sim_own]
            for i, (ref_mask, mask_weight) in enumerate(zip(self.ref_masks, self.mask_weights)):
                ref_mask = self.get_ref_mask(ref_mask, mask_weight, H, W)
                sim_ref = sim_refs[..., H*W*i: H*W*(i+1)]
                sim_ref = sim_ref + ref_mask.masked_fill(ref_mask == 0, torch.finfo(sim.dtype).min)
                sim_or.append(sim_ref)
            sim = torch.cat(sim_or, dim=-1)
        attn = sim.softmax(-1)
        attn = attn.to(v.dtype)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        # print(attn.dtype, v.dtype)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out  

    def sa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Original self-attention forward function
        """
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        # print("标准",out.shape)
        return out
    
    def mrsa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Mutli-reference self-attention(MRSA) forward function
        """
        #! 如果是交叉注意力、当前步数不在指定范围内，或当前层不在指定层索引中
        #! 则直接使用标准的自注意力，不应用MRSA
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        B = q.shape[0] // num_heads // 2
        # print(q.shape, k.shape, v.shape) #torch.Size([32, 4096, 40]) torch.Size([32, 4096, 40]) torch.Size([32, 4096, 40])

        # 检查是否有足够的图像进行MRSA
        if B <= 1:
            # print("[DEBUG] No reference images available, using standard attention")
            return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        
        #! 分离有无条件的qkv
        #16 4096 40
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        #! 目标图像与参考图像分离
        #  The first batch is the q,k,v feature of $z_t$ (own feature), and the subsequent batches are the q,k,v features of $z_t^'$ (reference featrue)
        qu_o, qu_r = qu[:num_heads], qu[num_heads:] 
        qc_o, qc_r = qc[:num_heads], qc[num_heads:]
        
        ku_o, ku_r = ku[:num_heads], ku[num_heads:]
        kc_o, kc_r = kc[:num_heads], kc[num_heads:]
        
        vu_o, vu_r = vu[:num_heads], vu[num_heads:]
        vc_o, vc_r = vc[:num_heads], vc[num_heads:]
        # print(num_heads, ku_r.shape, vu_r.shape) #8 torch.Size([8, 4096, 40]) torch.Size([8, 4096, 40])
        #! 拼接目标和参考的k和v
        ku_cat, vu_cat = torch.cat([ku_o, *ku_r.chunk(B-1)], 1), torch.cat([vu_o, *vu_r.chunk(B-1)], 1)
        kc_cat, vc_cat = torch.cat([kc_o, *kc_r.chunk(B-1)], 1), torch.cat([vc_o, *vc_r.chunk(B-1)], 1)

        #! MRSA注意力计算
        out_u_target = self.attn_batch(qu_o, ku_cat, vu_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
        out_c_target = self.attn_batch(qc_o, kc_cat, vc_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
        
        # The larger the style_fidelity, the more like the reference concepts, range of values: [0,1]
        if self.style_fidelity > 0:
            out_u_target = (1 - self.style_fidelity) * out_u_target + self.style_fidelity * self.attn_batch(qu_o, ku_o, vu_o, None, None, is_cross, place_in_unet, num_heads, **kwargs)

        out = self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        out_u, out_c = out.chunk(2)
        out_u_ref, out_c_ref = out_u[1:], out_c[1:]
        out = torch.cat([out_u_target, out_u_ref, out_c_target, out_c_ref], dim=0)
        # print("自定义",out.shape)
        return out