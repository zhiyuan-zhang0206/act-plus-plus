import tensorflow_datasets as tfds
import tensorflow as tf
import h5py
import os

class BimanualRT1(tfds.core.GeneratorBasedBuilder):
    """My dataset."""
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Description of your dataset."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(None, None, 3)),
                "label": tfds.features.ClassLabel(names=["class1", "class2"]),
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: Adjust file paths as needed
        hdf5_filepath = os.path.join(os.path.dirname(__file__), "my_dataset.hdf5")
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"filepath": hdf5_filepath},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with h5py.File(filepath, 'r') as f:
            images = f['images'][:]
            labels = f['labels'][:]
            for idx, (image, label) in enumerate(zip(images, labels)):
                yield idx, {
                    "image": image,
                    "label": label,
                }
