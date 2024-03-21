"""bimanual dataset."""

import tensorflow_datasets as tfds
import os
from pathlib import Path

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bimanual dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(bimanual): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
                'observation': {
                    'image_left': tfds.features.Image(),
                    'image_right': tfds.features.Image(),
                    'natural_language_instruction': tf.string,
                    'natural_language_embedding': tfds.features.Tensor(shape=(512,), dtype=tf.float32),
                },
                'is_terminal': tfds.features.bool,
                'is_last': tf.bool,
                'is_first': tf.bool,
                'action': {
                    'base_displacement_vector': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                    'base_displacement_vertical_rotation': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'gripper_closedness_action': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'rotation_delta': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    'world_vector': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                },
            }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('image', 'label'),  # Set to `None` to disable
        # homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(bimanual): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path = Path(os.environ['TFDS_DATA_DIR']) / 'bimanual'

    # TODO(bimanual): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train'),
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(bimanual): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
