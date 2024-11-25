import torch
from torchvision import transforms

import os
import sys

sys.path.append("lego-sorter/src")

from data import download
from model import engine, model
from common import utils, tools

print("hello")