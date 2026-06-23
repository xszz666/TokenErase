#The code for training is coming soon.

import argparse
import time
import logging
import math
import os
import random
import shutil
import warnings
import pandas as pd  
import csv 
import matplotlib.pyplot as plt  
from pathlib import Path
from typing import List, Optional

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from textsliders import prompt_util
from textsliders import train_util
from textsliders.prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings

#! 新加的包
from ref_util.util import load_image, load_mask, process_reference_images, load_reference_config, create_mrsa_from_references
from ref_util.hack_attention_integrated import hack_self_attention_to_mrsa


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available

