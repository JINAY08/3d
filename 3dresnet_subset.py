
import random
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    
    with VideoFileClip(video_path) as video:
        video_length = video.fps * video.duration

        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.uniform(0, max_start + 1)

        for _ in range(n_frames):
            time_pos = start + _ * frame_step
            frame = video.get_frame(time_pos / video.fps)
            frame = Image.fromarray((frame * 255).astype('uint8')).resize(output_size)
            frame = np.array(frame)[:, :, :3]
            result.append(frame)

    result = np.array(result)

    return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label