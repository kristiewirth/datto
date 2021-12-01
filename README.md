[![](https://api.travis-ci.com/kristiewirth/datto.svg?branch=master)](https://travis-ci.com/github/kristiewirth/datto)
[![](https://readthedocs.org/projects/datto/badge/)](https://datto.readthedocs.io/en/latest/)

# Installation

`pip install datto`

# Overview

datto is a package with various data tools to help in data analysis and data science work.

You can find the [documentation here](https://datto.readthedocs.io/en/latest/).

Some examples of what you can do:

- Remove links from some text
- Extract body of an email only (no greeting or signature)
- Easily load/save data from S3
- Run SQL from Python
- Explore data - check for mistyped data, find correlated data
- Assign a given user to an experimental condition
- Create an HTML dropdown from a DataFrame
- Find the most common phrases by a category
- Classify free text responses into any number of meaningful groups (e.g. find survey themes)
- Make a simple Python logger with default options
- Take some data and test various machine learning models on it

For detailed examples of how you can use it, check out [this Juypter notebook](datto_examples.ipynb).

# Other Templates

Check out the [templates folder](templates) for files that automate certain tasks, but don't fit within the realm of a Python package.

# Contributing

Create virtualenv:

```bash
pyenv virtualenv datto
```

Activate virtualenv:

```bash
pyenv activate datto
```

Install dependencies (specified in pyproject.toml file) in virtualenv:

```bash
poetry install
```

To add any new dependencies you need to Poetry, run:

```bash
poetry add PACKAGE_NAME
```

If adding/deleting a module, you'll need to update these files:

* `datto/__init__.py`
* `docs/source/index.rst`
* `docs/source/datto.rst`

Run the following to make sure all tests pass:

```bash
make test
```

Run the following to update all the docs:

```bash
make docs
```

Create a PR with your desired change(s), and request review from the code owner!
