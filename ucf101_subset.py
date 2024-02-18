import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import make_dataset, find_classes

class UCF101Dataset(VisionDataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
    ...

    Args:
        root (str): Root directory of the UCF101 Dataset.
        annotation_path (str): Path to the folder containing the split files.
        frames_per_clip (int): Number of frames in a clip.
        step_between_clips (int, optional): Number of frames between each clip.
        frame_rate (int, optional): Frame rate of videos (if known).
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        _precomputed_metadata (dict, optional): Precomputed metadata dictionary.
        num_workers (int, optional): Number of workers for data loading.
        _video_width (int, optional): Video frame width (if known).
        _video_height (int, optional): Video frame height (if known).
        _video_min_dimension (int, optional): Minimum video dimension (if known).
        _audio_samples (int, optional): Number of audio samples (if known).
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A 3-tuple with the following entries:
            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
               and `L` is the number of points
            - label (int): class of the video clip
    """

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC",
    ) -> None:
        super().__init__(root)
        extensions = ("avi",)

        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )
        self.video_clips = video_clips
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.video_clips.metadata

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label
