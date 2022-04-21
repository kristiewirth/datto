import os

from datto.DataConnections import NotebookConnections, S3Connections

s3 = S3Connections()
is_travis = "TRAVIS" in os.environ


nc = NotebookConnections()


def test_save_as_script():
    # Remove previous file if it exists to properly test function
    try:
        os.system("rm files_to_ignore/start_notebook.py")
    except Exception:
        pass

    nc.save_as_script("files_to_ignore/start_notebook.ipynb")

    saved_script = open("files_to_ignore/start_notebook.py").read()

    assert saved_script is not None


def test_save_as_notebook():
    # Remove previous file if it exists to properly test function
    try:
        os.system("rm files_to_ignore/start_script.ipynb")
    except Exception:
        pass

    nc.save_as_notebook("files_to_ignore/start_script.py")

    saved_notebook = open("files_to_ignore/start_script.ipynb").read()

    assert saved_notebook is not None
