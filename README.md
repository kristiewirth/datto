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
- Take some data and test a bunch of machine learning models on it

For detailed examples of how you can use it, check out [this Juypter notebook](datto_examples.ipynb).

# Other Templates

Check out the [templates folder](datto/tree/master/templates) for files and code snippets that automate certain tasks, but don't fit within the realm of a Python package.

Recommended: Copy the file contents into a [text expander app](https://zapier.com/blog/text-expander-how-to/) for easy future use.

# Contributing

Create virtualenv (specify version of Python you want):

```bash
pyenv virtualenv 3.6 datto
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

Run tests:

Run the following to make sure all tests pass:

```bash
make test
```

Submitting a change:

Create a PR with your desired change(s), and request review from the code owner!
