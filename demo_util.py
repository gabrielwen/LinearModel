from google.cloud import storage
from pathlib import Path
# Get some project config information
import json
import joblib
import re

import logging
import os
import re
import subprocess
import sys

from kubernetes import client, config

def notebook_setup():
    # Install some pip pckages
    # Install the SDK
    for p in ["feast", "retrying", "tensorflow", "tensorflow_data_validation"]:
        subprocess.check_call(["pip3", "install", p])
        
    fairing_code = os.path.join(Path.home(), "LinearModel", "fairing")

    if os.path.exists(fairing_code):    
        logging.info("Adding %s to path", fairing_code)
        sys.path = [fairing_code] + sys.path
        
    logging.basicConfig(format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)
    
    subprocess.check_call(["gcloud", "auth", "configure-docker", "--quiet"])
    subprocess.check_call(["gcloud", "auth", "activate-service-account", 
                           "--key-file=" + os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                           "--quiet"])
    
   
def get_project_config():
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    cred = {}
    with open(cred_path, 'r') as c:
        cred = json.load(c)

    PROJECT = cred['project_id']
    APP_NAME = re.search('([a-z\-]+)-user'.format(PROJECT),
                         cred['client_email']).group(1)
    p = subprocess.Popen(['gcloud', 'container', 'clusters', 'list',
                          '--filter', 'name=%s' % APP_NAME, '--format', 'json'],
                        stdout=subprocess.PIPE)
    
    out, _ = p.communicate()
    config = json.loads(out)[0]
    ZONE = config['zone']
    return PROJECT, ZONE, APP_NAME

def save_proto(data, path):
    """Save proto"""
    local_path = path
    
    is_gcs = False
    if path.startswith("gs://"):
        is_gcs = True
        local_path = "/tmp/" + os.path.basename(path)
    
    logging.info("Saving proto to %s", local_path)
    with open(local_path, "wb") as hf:
        hf.write(data.SerializeToString())
    if is_gcs:
        bucket_name, obj_path = split_gcs_uri(path)
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)   
        blob = bucket.blob(obj_path)
    
        logging.info("Uploading proto to %s", path)
        blob.upload_from_filename(local_path)

    logging.info("Saved proto to %s", path)

def save_df(df, path, key):
    local_path = path
    
    is_gcs = False
    if path.startswith("gs://"):
        is_gcs = True
        local_path = "/tmp/" + os.path.basename(path)
    
    df.to_hdf(local_path, key)
    logging.info("Saving DataFrame to %s; key %s", local_path, key)
    if is_gcs:
        bucket_name, obj_path = split_gcs_uri(path)
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)   
        blob = bucket.blob(obj_path)
    
        logging.info("Uploading dataframe to %s", path)
        blob.upload_from_filename(local_path)
        
def save_as_json(data, model_file):    
    gcs_path = None
    if model_file.startswith("gs://"):
        gcs_path = model_file        
        model_file = "/tmp/" + os.path.basename(model_file)


    logging.info("Saving data to: %s", model_file)
    
    with open(model_file, 'w+') as f:
        json.dump(data, f)
        
    if gcs_path:
        model_bucket_name, model_path = split_gcs_uri(gcs_path)
        storage_client = storage.Client()
        model_bucket = storage_client.get_bucket(model_bucket_name)   
        model_blob = model_bucket.blob(model_path)
    
        logging.info("Uploading data to %s", gcs_path)
        model_blob.upload_from_filename(model_file)
        

def split_gcs_uri(gcs_uri):
    """Split a GCS URI into bucket and path."""
    GCS_REGEX = re.compile("gs://([^/]*)(/.*)?")
    m = GCS_REGEX.match(gcs_uri)
    bucket = m.group(1)
    path = ""
    if m.group(2):
        path = m.group(2).lstrip("/")
    return bucket, path
       