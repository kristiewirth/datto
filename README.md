# Installation

`pip install datto`

# Overview

datto is a package with various data tools to help in data analysis and data science work.

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

# Example usage

```python
from datto.CleanText import CleanText

text = "I am some text with a link: www.google.com."

ct = CleanText()
cleaned_text = ct.remove_links(text)
```

# Contents

_(Updated 6/23/20)_

### CleanText

- clean_column_names()
- remove_links()
- fix_col_data_type()
- remove_duplicate_columns()
- lematize()
- remove_email_greetings_signatures()

### DataConnections

- load_from_s3()
- save_to_s3()
- setup_redshift_connection()
- run_sql_redshift()
- KafkaInterface

### Eda

- check_for_mistyped_booleans()
- find_cols_to_exclude()
- sample_unique_vals()
- find_correlated_features()
- separate_cols_by_type()

### Experiments

- assign_condition_by_id()

### FrontEnd

- dropdown_from_dataframe()

### ModelResults

- coefficients_summary()
- most_common_words_by_group()
- score_final_model()
- most_similar_texts()

### Setup

- display_more_data()
- setup_logger()

### TrainModel

- model_testing()
- train_test_split_by_ids()
