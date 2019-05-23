#!/usr/bin/env python
# coding: utf-8

# ## Taxi Ride Fare Prediction Using Kubeflow, Feast, and TFX
# 
# * Predict taxi ride fares using Feast and Kubeflow

# Setup the notebook
# - Install `feast` with pip.
# - Activate user service account with credentials JSON.
# - Hacks to retrieve essential information for deployments and serving.
# 
# **NOTE**: This code block might hangs for a long time.

# fairing:include-cell
from google.cloud import storage
import datetime
import demo_util
import kfp
import kfp.components as comp
import kfp.gcp as gcp
import kfp.dsl as dsl
import kfp.compiler as compiler
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
import tensorflow as tf
import tensorflow_data_validation as tfdv

# fairing:include-cell
class TaxiFeast(object):
    """Taxi code."""
    # Connect to the Feast deployment
    FEAST_CORE_URL = '10.148.0.99:6565'
    FEAST_SERVING_URL = '10.148.0.100:6566'
    STAGING_LOCATION = 'gs://kubecon-19-gojek/staging'
    
    STATS_KEY = "training_stats"

    def __init__(self):
        self._fs = None
        self._train_feature_set = None
    @property
    def fs(self):
        if not self._fs:
            self._fs = Client(core_url=self.FEAST_CORE_URL,serving_url=self.FEAST_SERVING_URL, verbose=True)
        return self._fs
    
    def compute_training_stats(self, stats_path=None):
        """compute training stats."""
        dataset = self.fs.create_dataset(self.train_feature_set, "2009-01-01", "2016-01-01")
        training_df = self.fs.download_dataset_to_df(dataset, self.STAGING_LOCATION)
        training_stats = tfdv.generate_statistics_from_dataframe(training_df)
        
        if stats_path:
            logging.info("Saving training stats to %s", stats_path)
            demo_util.save_proto(training_stats, stats_path)
        else:
            logging.info("No stats_path provided; not saving stats")
        return training_stats
    
    @property
    def train_feature_set(self):
        if not self._train_feature_set:
            self._train_feature_set = FeatureSet(entity=ENTITY_ID, 
                                                 features=TRAINING_FEATURES_SET)
        return self._train_feature_set
            

# ## Load raw data

# ## Extract more features

# ## Register entity and features

# ## Define a Feature Set for this project

# fairing:include-cell
ENTITY_ID = 'taxi_ride'
TRAINING_FEATURES_SET = [
    'taxi_ride.passenger_count',
    'taxi_ride.distance_haversine',
    'taxi_ride.distance_dummy_manhattan',
    'taxi_ride.direction',
    'taxi_ride.month',
    'taxi_ride.day_of_month',
    'taxi_ride.day_of_week',
    'taxi_ride.hour',
    'taxi_ride.fare_amount'
]

# ## Retrieve a Training Set from Feast

# ## Visualize statistics with TFDV

# ## Train Linear Model

# fairing:include-cell
class TaxiRideModel(TaxiFeast):
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
    super(TaxiRideModel, self).__init__()
    self.m = None
    self.b = None
    self.serving_fs = None

    logging.basicConfig(level=logging.INFO,
        format=('%(levelname)s|%(asctime)s'
                '|%(pathname)s|%(lineno)d| %(message)s'),
        datefmt='%Y-%m-%dT%H:%M:%S',
        )
    logging.getLogger().setLevel(logging.INFO)

  # Train model 
  def train(self, training_df, model_path):
    np.set_printoptions(precision=3)
    train_data = training_df[[x.split('.')[1] for x in TRAINING_FEATURES_SET]].to_numpy()
    train_data[:, len(train_data[0]) - 1] = 1
    Y = training_df['fare_amount'].to_numpy()

    x = np.linalg.lstsq(train_data, Y, rcond=0)[0]
    m, b = x[:len(train_data[0])-1], x[len(train_data[0])-1]

    self.m = m
    self.b = b
    
    self.save_model(model_path)
    
    return m,b

  def train_on_time_range(self, start_day, end_day, model_path):
    dataset = self.fs.create_dataset(self.train_feature_set, start_day, end_day)
    training_df =  self.fs.download_dataset_to_df(dataset, self.STAGING_LOCATION)
    return self.train(training_df, model_path)
    
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
        'FEAST_CORE_URL': self.FEAST_CORE_URL,
        'FEAST_SERVING_URL': self.FEAST_SERVING_URL,
        'ENTITY_ID': ENTITY_ID,
    }
    
    logging.info('Saving model to %s', model_path)

    demo_util.save_as_json(model, model_path)
  
  def preprocess(self):
    pass

  def validate(self):
    pass

# ## Train Locally

# ## Local Prediction

# ## Train and Deploy on Kubernetes

# ### Use fairing to build the docker image
# 
# * This uses the append builder to rapidly build docker images

# ### Launch a K8s job to compute the stats

# ## Deploy with Kubeflow

# ## Call the prediction endpoint

# ## Pipelines


if __name__ == "__main__":
  import fire
  import logging
  logging.basicConfig(format='%(message)s')
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(TaxiRideModel)
