import datto as dt
import pandas as pd
import spacy
from featurestore import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from spacy.cli import download

# Internal package for Zapier for running SQL `pip install -i https://pypi.zapier.com/simple/ featurestore`

# Need to load in English language information into spacy the first time you run this
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def lematize(text: str):
    # Lematizing means standardizing words to their root words, because they probably have similar meanings
    # E.g. jumping -> jump / so that texts with jumping are listed the same as texts with jump
    # More info: https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/
    spacy_text = nlp(text)
    return [token.lemma_ for token in spacy_text if not token.is_space]


# To use fs, you'll need to have the following variables defined in your .bash_profile file
# RS_DBNAME
# RS_HOST
# RS_PORT
# RS_USER
# RS_PASSWORD
fs = FeatureStore()

# Get data
# Non-Zapier equivalent is pd.read_sql
df = fs.read_redshift("""SELECT my_data FROM some_table""")
# https://docs.looker.com/sharing-and-publishing/downloading
# df = pd.read_csv("some_data_I_downloaded_maybe_looker_data.csv")

# Fill null values by type
df_nums = df.select_dtypes("number")
df[df_nums.columns] = df_nums.fillna(0)
df.fillna("", inplace=True)

###### CLASSIFICATION MODELS ######
# Models where you're predicting a category, yes/no

# Language processing - you can omit this step if doing a simpler problem
# E.g. using number of tasks for an account to predict upgrade or not

# Look into adding domain specific stop words as well as pre-built stop words
# Stop words -> words that are too common / not meaningful, so remove these from the dataset first
stop_words = set(list(ENGLISH_STOP_WORDS) + ["Zapier", "Zap"])

# Vectorize test (i.e. make adjusted count columns)
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    tokenizer=lematize,
    # Not just word columns (1), also make columns of phrases (2 or 3 word phrases)
    ngram_range=(1, 3),
    # Means each word must appear in at least __ different texts to be included
    min_df=2,
    # Means each word can't appear in more than _% of texts to be included
    max_df=0.4,
)
vectors = vectorizer.fit_transform(df["my_text_column"])

# Changing vectors back to a pandas DataFrame (easier to work with, preserves text column names)
vectors = pd.DataFrame(vectors.todense())

# Setting the word/phrase tokens as the column names
words = vectorizer.get_feature_names()
vectors.columns = words
cleaned_df = pd.concat([df, vectors], axis=1)

# Split into input data vs target (what you're predicting)
X = cleaned_df.drop("label_for_text", axis=1)
y = df["label_for_test"]

# Split further into training set (model learns from this)
# And testing set (model hasn't seen this, you can use it to make predictions and see how well it's doing by comparing predictions vs actual labels)
# This is called cross validation: https://machinelearningmastery.com/k-fold-cross-validation/
X_train, X_test, y_train, y_test = train_test_split(X, y)

# If you have any feature columns with categorical data (e.g. user role), you need to make these into dummy variables instead
# Basically all columns need to be numbers, you can't feed in text data
# You want to base this off your training set so that there isn't knowledge leaking into your test set (i.e. the model is cheating)
# Don't need to do this to text data you already vectorized -> it's already represented as numbers in your data
# More info: https://www.askpython.com/python/examples/creating-dummy-variables
X_train_dummies = pd.get_dummies(X_train)
X_test_dummies = X_test.reindex(columns=X_train_dummies.columns, fill_value=0)

# Train model on data
# Instantiate model
# This is just one common type of model, there's a whole bunch of models just within sklearn
# Available sklearn models: https://scikit-learn.org/stable/supervised_learning.html
model = RandomForestClassifier()
model.fit(X_train_dummies, y_train)

# Predict on your unseen test data
y_predicted = model.predict(X_test_dummies)

# See how well your model is doing
# Accuracy is a common measurement, but can be misleading
# Precision & recall can be a better summary of how well your model is doing
# Precision: out of everything I predicted to be spam, what percent is actually spam?
# Recall: out of all the spam text in the data, what percent did I catch (i.e. predicted as spam?)
# More info: https://towardsdatascience.com/whats-the-deal-with-accuracy-precision-recall-and-f1-f5d8b4db1021
print(accuracy_score(y_test, y_predicted))
print(precision_score(y_test, y_predicted))
print(recall_score(y_test, y_predicted))

###### REGRESSION MODELS ######
# Models where you're predicting a number
# E.g. predicting number of Zaps a user will activate

# Split into X, y as above
# Split into train and test sets
# Instead of a classifier, use a regression type model
# Make predictions as above
# Will need to measure different scoring metrics
print(mean_squared_error(y_test, y_predicted))
print(median_absolute_error(y_test, y_predicted))
print(r2_score(y_test, y_predicted))

###### datto METHODS ######
# Above are the basic steps, but I have lots of methods in my Python package that do similar things/simplify things too!
# https://github.com/kristiewirth/datto
# It can do way more stuff, but here's a few examples I use often

mr = dt.ModelResults()

# More detailed model results
mr.score_final_model("classification", X_test_dummies, y_test, model)
mr.score_final_model("regression", X_test_dummies, y_test, model)

# See coefficients for model, i.e. how each feature (piece of information) influences the predictions / which are most important
mr.coefficients_graph(
    X_train_dummies, X_test_dummies, model, "classification", "filename"
)

# See some example individual predictions
mr.coefficients_individual_predictions(
    model,
    X_train_dummies,
    X_train_dummies,
    X_test_dummies,
    "name_of_unique_id_per_row_like_account_id",
    num_id_examples=5,
    num_feature_examples=10,
    model_type="classification",
    class_names=y_test.columns,
)
