import os

from datto.DataConnections import NotebookConnections, S3Connections

s3 = S3Connections()
is_travis = "TRAVIS" in os.environ

os.system("mkdir -p tests/temp_files")

nc = NotebookConnections()


def test_save_as_script():
    # Create dummy file to test conversion
    os.system("""echo 'print("Hello, world!")' > tests/temp_files/start_notebook.py""")
    os.system(f"jupytext --to notebook tests/temp_files/start_notebook.py")

    nc.save_as_script("tests/temp_files/start_notebook.ipynb")

    saved_script = open("tests/temp_files/start_notebook.py").read()

    assert saved_script is not None

    os.system("rm tests/temp_files/*")


def test_save_as_notebook():
    # Create dummy file to test conversion
    os.system("""echo 'print("Hello, world!")' > tests/temp_files/start_script.py""")

    nc.save_as_notebook("tests/temp_files/start_script.py")

    saved_notebook = open("tests/temp_files/start_script.ipynb").read()

    assert saved_notebook is not None

    os.system("rm tests/temp_files/*")
