# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper
from .build import *

from .datasets import *
from .ytvis_eval import YTVISEvaluator
from .ovis_eval import OVISEvaluator
from .lvvis_eval import LVVISEvaluator
from .lvvis_eval_video import LVVISEvaluator_video
from .burst_eval import BURSTEvaluator
