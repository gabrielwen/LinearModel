from pathlib import Path
# Get some project config information
import json
import re

import logging
import os
import subprocess
import sys

from kubernetes import client, config

def notebook_setup():
    # Install some pip pckages
    # Install the SDK
    for p in ["feast", "retrying"]:
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


def get_fairing_endpoint(serving_label):
    config.load_kube_config()
    c = client.Configuration()
    client.Configuration.set_default(c)

    v1 = client.CoreV1Api()
    body = client.V1Service()
    label_selector = 'serving=%s' % serving_label
    resp = v1.list_service_for_all_namespaces(label_selector=label_selector)

    service_name = resp.items[0].metadata.name
    namespace = resp.items[0].metadata.namespace

    print('fairing service: {0}/{1}'.format(namespace, service_name))
    return namespace, service_name