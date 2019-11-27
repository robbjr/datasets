"""TODO(BCCD): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import xml.etree.ElementTree

_VOC_POSES=['unspecified']
_VOC_LABELS=['rbc', 'wbc', 'platelets', 'platetets', 'platlets', 'platetlets']
_HEIGHT = 480
_WIDTH = 640

# TODO(BCCD): BibTeX citation
_CITATION = """
https://github.com/Shenggan/BCCD_Dataset
"""
_URL = "https://storage.googleapis.com/duke-tfds/bccd/bccd_data.tar.gz"
# TODO(BCCD):
_DESCRIPTION = """
"""
class Bccd(tfds.core.GeneratorBasedBuilder):
  """TODO(BCCD): Short description of my dataset."""
  VERSION = tfds.core.Version('2.0.0')

  def _get_example_objects(self, annon_filepath):
    """Function to get all the objects from the annotation XML file."""
    with tf.io.gfile.GFile(annon_filepath, "r") as f:
      root = xml.etree.ElementTree.parse(f).getroot()

      for obj in root.findall("object"):
        # Get object's label name.
        label = obj.find("name").text.lower()
        # Get objects' pose name.
        pose = obj.find("pose").text.lower()
        is_truncated = (obj.find("truncated").text == "1")
        is_difficult = (obj.find("difficult").text == "1")
        bndbox = obj.find("bndbox")
        xmax = float(bndbox.find("xmax").text)
        xmin = float(bndbox.find("xmin").text)
        ymax = float(bndbox.find("ymax").text)
        ymin = float(bndbox.find("ymin").text)
        yield {
            "label": label,
            "pose": pose,
            "bbox": tfds.features.BBox(ymin / _HEIGHT, xmin / _WIDTH, ymax / _HEIGHT, xmax / _WIDTH),
            "is_truncated": is_truncated,
            "is_difficult": is_difficult,
        }

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'xml_filepath': tfds.features.Tensor(shape=(), dtype=tf.string),
            'objects': tfds.features.Sequence({
                'label': tfds.features.ClassLabel(names=_VOC_LABELS),
                'bbox': tfds.features.BBoxFeature(),
                'pose': tfds.features.ClassLabel(names=_VOC_POSES),
                'is_truncated': tf.bool,
                'is_difficult': tf.bool,
            })
        }),
        supervised_keys=('image_annotation','image_name'),
        citation=_CITATION
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract({
        'data': _URL
    })    
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir": '{}/bccd_data/images'.format(path['data']),
                "annotations_dir": '{}/bccd_data/annotations'.format(path['data'])
            },
        )
    ]

  def _generate_examples(self, images_dir, annotations_dir):
    """Yields examples."""    
    for filename in tf.io.gfile.listdir(images_dir):
        if filename.split('_')[0] != 'BloodImage':
            continue
            
        image_filepath = '{}/{}'.format(images_dir, filename)
        xml_filepath = '{}/{}.xml'.format(annotations_dir, filename.split('.')[0])
        
        if not tf.io.gfile.exists(image_filepath):
            continue
        
        if not tf.io.gfile.exists(xml_filepath):
            continue
        
        objects = list(self._get_example_objects(xml_filepath))
        yield filename, {
            "image": image_filepath,
            "xml_filepath": xml_filepath,
            "objects": objects
        }
