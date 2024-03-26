"""bimanual dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
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
    return tfds.core.DatasetInfo(
            builder=self,
            description="Your dataset description goes here.",
            features=tfds.features.FeaturesDict({
                'observation': tfds.features.FeaturesDict({
                    'image_left': tfds.features.Sequence(tfds.features.Image(shape=(300, 300, 3)), length=15),
                    'image_right': tfds.features.Sequence(tfds.features.Image(shape=(300, 300, 3)), length=15),
                    'natural_language_instruction': tf.string,
                    'natural_language_embedding': tfds.features.Sequence(tfds.features.Tensor(shape=(512,), dtype=tf.float32), length=15),
                }),
                'is_terminal': tfds.features.Sequence(tf.bool, length=15),
                'is_last': tfds.features.Sequence(tf.bool, length=15),
                'is_first': tfds.features.Sequence(tf.bool, length=15),
                'action': tfds.features.FeaturesDict({
                    'base_displacement_vector': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                    'base_displacement_vertical_rotation': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'gripper_closedness_action': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'rotation_delta': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    'world_vector': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                }),
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
    path = Path(os.environ['TFDS_DATA_DIR']) / 'bimanua_zzy' / '0.1.0'

    # TODO(bimanual): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train'),
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(bimanual): Yields (key, example) tuples from the dataset
    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
    for f in path.glob('*.npy'):
      data = np.load(f).item()
      replicated_embedding = np.tile(data['language_embedding'], (15, 1))
            
      # Create arrays of length 15 filled with the same boolean value
      is_terminal_array = np.full(15, data.get('is_terminal', False))
      is_last_array = np.full(15, data.get('is_last', False))
      is_first_array = np.full(15, data.get('is_first', False))
      
      yield {
          'observation': {
              'image_left': data['left_image'][i:i+15],  # Sequence of 15 images
              'image_right': data['right_image'][i:i+15],  # Sequence of 15 images
              'natural_language_instruction': data['language_instruction'],  # Single instruction for all
              'natural_language_embedding': replicated_embedding,  # Replicated embedding 15 times
          },
          'is_terminal': is_terminal_array,  # Array of booleans
          'is_last': is_last_array,  # Array of booleans
          'is_first': is_first_array,  # Array of booleans
          'action': {
              'base_displacement_vector': data['base_displacement_vector'],
              'base_displacement_vertical_rotation': data['base_displacement_vertical_rotation'],
              'gripper_closedness_action': data['gripper_closedness_action'],
              'rotation_delta': data['rotation_delta'],
              'world_vector': data['world_vector'],
          },
      }
