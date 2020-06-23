# Installation

`pip install datto`

# Overview

datto is a package with various data tools to help in data analysis and data science work.

# Contents

_(Updated 6/23/20)_

### CleanText

- clean_column_names()
- remove_links()
- fix_col_data_type()
- remove_duplicate_columns()
- remove_names()
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

- coefficents_summary()
- most_common_words_by_group()
- score_final_model()
- most_similar_texts()

### Setup

- display_more_data()
- setup_logger()

### TrainModel

- model_testing()
- train_test_split_by_ids()
