import numpy as np
import json
import logging

from feast.sdk.resources.feature_set import FeatureSet
from feast.sdk.client import Client

class LabelPrediction(object):
  def __init__(self):
    self.m = None
    self.b = None
    self.fs = None
    self.serving_fs = None

    logging.basicConfig(level=logging.INFO,
        format=('%(levelname)s|%(asctime)s'
                '|%(pathname)s|%(lineno)d| %(message)s'),
        datefmt='%Y-%m-%dT%H:%M:%S',
        )
    logging.getLogger().setLevel(logging.INFO)

  def predict(self, feature_id, feature_names):
    logging.info('feature_id = %s', feature_id)
    logging.info('feature_names = %s', feature_names)
    if any([i is None for i in [self.m, self.b, self.fs, self.serving_fs]]):
      with open('simple_model.dat', 'r') as f:
        model = json.load(f)
        self.m = np.array(model.get('m', []))
        self.b = float(model.get('b', 0))

        _FEAST_CORE_URL = model['FEAST_CORE_URL']
        _FEAST_SERVING_URL = model['FEAST_SERVING_URL']
        _ENTITY_ID = model['ENTITY_ID']
        _FEATURES_SET = model['FEATURES_SET']

        logging.info('FEAST_CORE_URL: %s', _FEAST_CORE_URL)
        logging.info('FEAST_SERVING_URL: %s', _FEAST_SERVING_URL)
        logging.info('ENTITY_ID: %s', _ENTITY_ID)
        logging.info('FEATURES_SET: %s', _FEATURES_SET)

        self.fs = Client(core_url=_FEAST_CORE_URL,
            serving_url=_FEAST_SERVING_URL,
            verbose=True)
        self.serving_fs = FeatureSet(
            entity=_ENTITY_ID,
            features=_FEATURES_SET)

    features = self.fs.get_serving_data(
        self.serving_fs,
        entity_keys=[feature_id])
    X = features.to_numpy()[0][1:]
    logging.info('X: %s', str(X))

    return sum(self.m * X) + self.b
