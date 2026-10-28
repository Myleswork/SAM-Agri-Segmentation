import json
import unittest
from engine.serialization import json_safe

class SerializationTests(unittest.TestCase):
 def test_non_finite_numbers_become_json_null(self):
  obj=json_safe({'a':[float('nan'),float('inf'),1.]})
  self.assertEqual(obj,{'a':[None,None,1.]})
  json.dumps(obj,allow_nan=False)
