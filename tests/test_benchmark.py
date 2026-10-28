import unittest
from engine.benchmark import latency_summary

class BenchmarkTests(unittest.TestCase):
 def test_latency_and_throughput(self):
  x=latency_summary([10,20,30,40],4)
  self.assertEqual(x['mean_ms'],25)
  self.assertAlmostEqual(x['images_per_second'],160)
  self.assertAlmostEqual(x['p50_ms'],25)
  self.assertEqual(x['batch_size'],4)
