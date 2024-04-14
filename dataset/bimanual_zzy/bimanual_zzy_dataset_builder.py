"""bimanual dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bimanual_zzy dataset."""

    VERSION = tfds.core.Version("0.1.1")
    RELEASE_NOTES = {
        "0.1.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(bimanual): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description="Your dataset description goes here.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image_left": tfds.features.Image(shape=(300, 300, 3)),
                                "image_right": tfds.features.Image(shape=(300, 300, 3)),
                                "natural_language_instruction": tfds.features.Text(),
                                "natural_language_embedding": tfds.features.Tensor(
                                    shape=(512,), dtype=tf.float32
                                ),
                            },
                            "is_terminal": tfds.features.Tensor(
                                shape=(), dtype=tf.bool
                            ),
                            "is_last": tfds.features.Tensor(shape=(), dtype=tf.bool),
                            "is_first": tfds.features.Tensor(shape=(), dtype=tf.bool),
                            "action": {
                                "base_displacement_vector": tfds.features.Tensor(
                                    shape=(2,), dtype=tf.float32
                                ),
                                "base_displacement_vertical_rotation": tfds.features.Tensor(
                                    shape=(1,), dtype=tf.float32
                                ),
                                "gripper_closedness_action_left": tfds.features.Tensor(
                                    shape=(1, ), dtype=tf.float32
                                ),
                                "gripper_closedness_action_right": tfds.features.Tensor(
                                    shape=(1, ), dtype=tf.float32
                                ),
                                "rotation_delta_left": tfds.features.Tensor(
                                    shape=(3,), dtype=tf.float32
                                ),
                                "world_vector_left": tfds.features.Tensor(
                                    shape=(3,), dtype=tf.float32
                                ),
                                "rotation_delta_right": tfds.features.Tensor(
                                    shape=(3,), dtype=tf.float32
                                ),
                                "world_vector_right": tfds.features.Tensor(
                                    shape=(3,), dtype=tf.float32
                                ),
                                "terminate_episode": tfds.features.Tensor(
                                    shape=(3,), dtype=tf.float32
                                ),
                            },
                        }
                    )
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(bimanual): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')
        path = Path(os.environ["TFDS_DATA_DIR"]) / "bimanual_zzy" / "data"
        # breakpoint()
        # TODO(bimanual): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train"),
            "test": self._generate_examples(path / "test"),
        }

    def _generate_examples(self, path: Path):
        """Yields examples."""
        # TODO(bimanual):
        # print(list(path.glob("*.npy")))
        # breakpoint()
        for f in path.glob("*.npy"):
            data = np.load(f, allow_pickle=True).item()
            language_embedding = data["language_embedding"]
            length = len(data["image_left"])
            episode = []
            terminate_episode_array = np.zeros((length, 3), dtype=np.float32)
            terminate_episode_array[:-2, 1] = 1
            terminate_episode_array[-2, 0] = 1
            for i in range(length):
                
                episode.append(
                    {
                        "observation": {
                            "image_left": data["image_left"][i] ,
                            "image_right":data["image_right"][i] ,
                            "natural_language_instruction": data["language_instruction"],
                            "natural_language_embedding": language_embedding.astype(np.float32),
                        },
                        "is_first": i == 0,
                        "is_last": i == length - 1,
                        "is_terminal": i == length - 1,
                        "action": {
                            "base_displacement_vector": np.zeros((2,), dtype=np.float32),
                            "base_displacement_vertical_rotation": np.zeros((1,), dtype=np.float32),
                            "gripper_closedness_action_left": data["gripper_closedness_action_left"][i].astype(np.float32).reshape((1, )),
                            "gripper_closedness_action_right": data["gripper_closedness_action_right"][i].astype(np.float32).reshape((1, )),
                            "rotation_delta_left": data["rotation_delta_left"][i].astype(np.float32),
                            "world_vector_left": data["world_vector_left"][i].astype(np.float32),
                            "rotation_delta_right": data["rotation_delta_right"][i].astype(np.float32),
                            "world_vector_right": data["world_vector_right"][i].astype(np.float32),
                            "terminate_episode": terminate_episode_array[i]
                        },
                    }
                )
            yield f.stem, {"steps": episode}
