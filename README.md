# Simple example to use Kubeflow for model training and deployments.

## Deploy Kubeflow cluster
1. Download `kfctl` CLI (v0.5.1) from [kubeflow release](https://github.com/kubeflow/kubeflow/releases)
1. Run the following command to deploy Kubeflow:
```bash
# Init using HEAD of v0.5-branch.
# This is needed because v0.5.1 doesn't include this fix:
# https://github.com/kubeflow/kubeflow/pull/3238
kfctl init {APP_NAME} --platform gcp --project {PROJECT} -V --version v0.5-branch

cd {APP_DIR}

kfctl generate all -V

kfctl apply all -V
```

## Notebook settings
1. Follow instructions on setting up notebook with UI: [link](https://www.kubeflow.org/docs/notebooks/setup/)
1. upload `Linear_Model.ipynb`/`deploy_with_fairing.py`/`LabelPrediction.py` to notebook.

## Misc
- `BASE_IMAGE` is built with `fairing_job/`. `Dockerfile` in this folder has minimum required dependencies for fairing service.
