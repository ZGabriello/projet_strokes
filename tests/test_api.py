import unittest
import requests

class ApiTest(unittest.TestCase):
  API_URL = "http://127.0.0.1:5000"
  status_url = "{}/status".format(API_URL)
  modele_url = "{}/modele".format(API_URL)
  performance_url = "{}/performance".format(API_URL)
  def test_get_status(self):
      r = requests.get(ApiTest.status_url)
      self.assertEqual(r.status_code, 200)
  def test_get_modele(self):
      r = requests.get(ApiTest.modele_url)
      self.assertEqual(r.status_code, 200)
  def test_get_performance(self):
      r = requests.get(ApiTest.performance_url)
      self.assertEqual(r.status_code, 200)
