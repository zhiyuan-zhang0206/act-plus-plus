"""bimanual dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bimanual dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(bimanual): Specifies the tfds.core.DatasetInfo object
        return (
            tfds.core.DatasetInfo(
                builder=self,
                description="Your dataset description goes here.",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": {
                                    "image_left": tfds.features.Image(
                                        shape=(300, 300, 3)
                                    ),
                                    "image_right": tfds.features.Image(
                                        shape=(300, 300, 3)
                                    ),
                                    "natural_language_instruction": tfds.features.Text(),
                                    "natural_language_embedding": tfds.features.Tensor(
                                        shape=(512,), dtype=tf.float32
                                    ),
                                },
                                "is_terminal": tfds.features.Tensor(
                                    shape=(), dtype=tf.bool
                                ),
                                "is_last": tfds.features.Tensor(
                                    shape=(), dtype=tf.bool
                                ),
                                "is_first": tfds.features.Tensor(
                                    shape=(), dtype=tf.bool
                                ),
                                "action": {
                                    "base_displacement_vector": tfds.features.Tensor(
                                        shape=(2,), dtype=tf.float32
                                    ),
                                    "base_displacement_vertical_rotation": tfds.features.Tensor(
                                        shape=(1,), dtype=tf.float32
                                    ),
                                    "gripper_closedness_action": tfds.features.Tensor(
                                        shape=(1,), dtype=tf.float32
                                    ),
                                    "rotation_delta": tfds.features.Tensor(
                                        shape=(3,), dtype=tf.float32
                                    ),
                                    "world_vector": tfds.features.Tensor(
                                        shape=(3,), dtype=tf.float32
                                    ),
                                },
                            }
                        )
                    }
                ),
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(bimanual): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')
        path = Path(os.environ["TFDS_DATA_DIR"]) / "bimanua_zzy" / "0.1.0"

        # TODO(bimanual): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train"),
            "test": self._generate_examples(path / "test"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(bimanual): 
        for f in path.glob("*.npy"):
            data = np.load(f).item()
            language_embedding = data["language_embedding"]
            length = len(data["left_image"])
            episode = []
            for i in range(length):
                episode.append({
                    "observation": {
                        "image_left": data["left_image"][i],
                        "image_right": data["right_image"][i],
                        "natural_language_instruction": data["language_instruction"],
                        "natural_language_embedding": language_embedding,
                    },
                    "is_first": data.get("is_first", i ==0),
                    "is_last": data.get("is_last", i == length - 1),
                    "is_terminal": data.get("is_terminal", i == length - 1),
                    "action": {
                        "base_displacement_vector": tf.zeros((2,), dtype=tf.float32),
                        "base_displacement_vertical_rotation": tf.zeros((1,), dtype=tf.float32),
                        "gripper_closedness_action_left": data["gripper_closedness_action_left"][i],
                        "gripper_closedness_action_right": data["gripper_closedness_action_right"][i],
                        "rotation_delta_left": data["rotation_delta"][i],
                        "world_vector_left": data["world_vector"][i],
                        "rotation_delta_right": data["rotation_delta_right"][i],
                        "world_vector_right": data["world_vector_right"][i],
                    },
                })
            yield f.stem, {"steps" : episode}
                    