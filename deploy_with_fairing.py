import argparse
import logging
import fairing
import os
import shutil
import tempfile
import fnmatch

def deploy(registry, base_image, serving_label=None):
  fairing.config.set_builder('append', registry=registry, base_image=base_image)
  labels = {
      "app": "mlapp",
      "purpose": "garielwen-test",
  }
  if not serving_label is None:
      labels["serving"] = serving_label
  fairing.config.set_deployer('serving', serving_class='LabelPrediction',
          labels=labels)

  this_dir = os.path.dirname(__file__)
  base_dir = os.path.abspath(os.path.join(this_dir, ".."))
  flask_dir = os.path.join(base_dir, "flask_app")
  input_files = []

  context_dir = tempfile.mkdtemp()

  for dir_to_copy in [flask_dir, this_dir]:
    for root, dirs, files in os.walk(dir_to_copy, topdown=False):
      for name in files:
        if not fnmatch.fnmatch(name, "*.py") and \
            not fnmatch.fnmatch(name, "*.dat"):
          continue
        shutil.copyfile(os.path.join(root, name),
                        os.path.join(context_dir, name))
        input_files.append(name)

  os.chdir(context_dir)
  fairing.config.set_preprocessor('python', input_files=input_files)
  return fairing.config.run()

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--registry", default="", type=str,
    help=("The registry where your images should be pushed"))

  parser.add_argument(
    "--base_image", default="", type=str,
    help=("The base image to use"))

  args = parser.parse_args()
  deploy(args.registry, args.base_image)
