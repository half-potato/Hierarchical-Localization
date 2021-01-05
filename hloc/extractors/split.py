import os

import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import sys

from pathlib import Path

from ..utils.base_model import BaseModel
from ..utils.base_model import dynamic_load
from .. import extractors as base

class Split(BaseModel):

    def _init(self, config):
        Detector = dynamic_load(base, config["detector"])
        Descriptor = dynamic_load(base, config["descriptor"])
        self.detector = Detector(config)
        self.descriptor = Descriptor(config)

    def _forward(self, data):
        output = self.detector._forward(data)
        output = self.descriptor._describe(data, output)
        return output

    # override
    def eval(self):
        self.descriptor = self.descriptor.eval()
        self.descriptor = self.descriptor.eval()
        return self

    # override
    def to(self, device):
        self.detector = self.detector.to(device)
        self.descriptor = self.descriptor.to(device)
        return self
