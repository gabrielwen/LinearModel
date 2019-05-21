#!/usr/bin/env python
# coding: utf-8

# ## Taxi Ride Fare Prediction Using Kubeflow and Feast
# 
# * Predict taxi ride fares using Feast and Kubeflow

# Setup the notebook
# - Install `feast` with pip.
# - Activate user service account with credentials JSON.
# - Hacks to retrieve essential information for deployments and serving.
# 
# **NOTE**: This code block might hangs for a long time.

# fairing:include-cell
import fairing
import sys
import importlib
import uuid
import logging
import os
import json
import requests
import pandas as pd
import numpy as np
from retrying import retry
from feast.sdk.resources.entity import Entity
from feast.sdk.resources.storage import Storage
from feast.sdk.resources.feature import Feature, Datastore, ValueType
from feast.sdk.resources.feature_set import FeatureSet, FileType
import feast.specs.FeatureSpec_pb2 as feature_pb

from feast.sdk.importer import Importer
from feast.sdk.client import Client

# ## Load raw data

# ## Extract more features

# ## Register entity and features

# ## Define a Feature Set for this project

# ## Retrieve a Training Set from Feast

# ## Train Linear Model

# fairing:include-cell
class TaxiRideModel(object):
  """Model class."""
  SERVING_FEATURE_SET = [
        'taxi_ride.passenger_count',
        'taxi_ride.distance_haversine',
        'taxi_ride.distance_dummy_manhattan',
        'taxi_ride.direction',
        'taxi_ride.month',
        'taxi_ride.day_of_month',
        'taxi_ride.day_of_week',
        'taxi_ride.hour']

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

  # Train model 
  def train(self, training_df):
    np.set_printoptions(precision=3)
    train_data = training_df[[x.split('.')[1] for x in TRAINING_FEATURES_SET]].to_numpy()
    train_data[:, len(train_data[0]) - 1] = 1
    Y = training_df['fare_amount'].to_numpy()

    x = np.linalg.lstsq(train_data, Y, rcond=0)[0]
    m, b = x[:len(train_data[0])-1], x[len(train_data[0])-1]

    self.m = m
    self.b = b
    return m,b

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

        logging.info('FEAST_CORE_URL: %s', _FEAST_CORE_URL)
        logging.info('FEAST_SERVING_URL: %s', _FEAST_SERVING_URL)
        logging.info('ENTITY_ID: %s', _ENTITY_ID)
        logging.info('FEATURES_SET: %s', self.SERVING_FEATURE_SET)

        self.fs = Client(core_url=_FEAST_CORE_URL,
            serving_url=_FEAST_SERVING_URL,
            verbose=True)
        self.serving_fs = FeatureSet(
            entity=_ENTITY_ID,
            features=self.SERVING_FEATURE_SET)

    features = self.fs.get_serving_data(
        self.serving_fs,
        entity_keys=[feature_id])
    X = features.to_numpy()[0][1:]
    logging.info('X: %s', str(X))

    return [sum(self.m * X) + self.b]

  def save_model(self, model_path):
    """Save the model to a json file."""
    MODEL_FILE = 'simple_model.dat'

    model = {
        'm': self.m.tolist(),
        'b': self.b,
        'FEAST_CORE_URL': FEAST_CORE_URL,
        'FEAST_SERVING_URL': FEAST_SERVING_URL,
        'ENTITY_ID': ENTITY_ID,
    }
    
    logging.info('Saving model to %s', model_path)

    with open(model_path, 'w+') as f:
        json.dump(model, f)

# ## Use fairing to build the docker image
# 
# * This uses the append builder to rapidly build docker images

# ## Local Prediction

# ## Save the model

# ## Deploy with Kubeflow

# ## Call the prediction endpoint


if __name__ == "__main__":
  import fire
  import logging
  logging.basicConfig(format='%(message)s')
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(TaxiRideModel)
