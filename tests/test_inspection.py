import unittest
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
from tools.inspect_dataset import indexed_files, check_band, split_intersections

class InspectionTests(unittest.TestCase):
    def test_case_insensitive_collision_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'a.JPG').touch(); (p / 'a.jpg').touch()
            with self.assertRaises(ValueError): indexed_files(p)

    def test_band_only_accepts_equal_channels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'band.tif'
            a = np.zeros((3, 4, 3), dtype=np.uint8)
            Image.fromarray(a).save(p)
            self.assertTrue(check_band(p)['channels_identical'])
            a[0, 0, 1] = 1; Image.fromarray(a).save(p)
            self.assertFalse(check_band(p)['channels_identical'])

    def test_split_intersection_ignores_extension_case(self):
        self.assertEqual(split_intersections({'train':['A.JPG'], 'val':['a.jpg']}), {'train:val':['a.jpg']})

class MetadataTests(unittest.TestCase):
    def test_red_edge_metadata_spelling(self):
        from tools.inspect_dataset import band_matches
        self.assertTrue(band_matches('RE', 'RedEdge'))
        self.assertTrue(band_matches('RE', 'Red Edge'))
        self.assertFalse(band_matches('RE', 'Red'))
