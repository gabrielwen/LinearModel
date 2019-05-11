import numpy as np
import json

class LabelPrediction(object):
  def __init__(self):
    self.m = None
    self.b = None

  def predict(self, X, feature_names):
    if self.m is None or self.b is None:
      with open('simple_model.dat', 'r') as f:
        model = json.load(f)
        self.m = np.array(model.get('m', []))
        self.b = float(model.get('b', 0))
    return self.m * X + self.b