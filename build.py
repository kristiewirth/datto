import subprocess

subprocess.call("pip install lightgbm --install-option=--nomp", shell=True)

subprocess.call('ARCHFLAGS="-arch x86_64" pip install shap', shell=True)
