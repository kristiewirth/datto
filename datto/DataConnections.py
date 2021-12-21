import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import psycopg2
import s3fs
from slack_sdk import WebClient


class S3Connections:
    """
    Interact with S3
    """

    def save_to_s3(
        self,
        directory_path,
        object_to_save,
        object_name,
    ):
        """
        Pickle and save an object to s3. Creates the folder specified if it does not yet exist.

        Parameters
        --------
        directory_path: str
            Starts with bucket name, slash any subdirectories
        object_to_save: any object with a type that can be pickled
        object_name: str

        Returns
        --------
        None
        """
        s3 = s3fs.S3FileSystem(anon=False)

        filepath = f"{directory_path}/{object_name}.pkl"

        try:
            with s3.open(filepath, "wb") as f:
                pickle.dump(object_to_save, f)
        except Exception:
            # If error, try creating folder
            s3.mkdir(f"{directory_path}/")

            with s3.open(filepath, "wb") as f:
                pickle.dump(object_to_save, f)

    def load_from_s3(self, directory_path, object_name):
        """
        Load a pickled object from s3.
        Note: The pickle module is not secure. Only unpickle data you trust/saved yourself.

        Parameters
        --------
        directory_path: str
            Starts with bucket name, slash any subdirectories
        object_name: str

        Returns
        --------
        saved_object
        """
        s3 = s3fs.S3FileSystem(anon=False)

        filepath = f"{directory_path}/{object_name}.pkl"

        saved_object = pickle.load(s3.open(filepath, mode="rb"))

        return saved_object


class SQLConnections:
    """
    Connect with a SQL database
    """

    def __init__(self, dbname=None, host=None, port=None, user=None, password=None):
        """
        Pandas doesn't integrate with Redshift directly. Instead use psycopg2 to connect.
        Pulls credentials from environment automatically if set.

        Parameters
        --------
        dbname: str
        host: str
        port: str
        user: str
        password: str

        Returns
        --------
        conn: cursor from database connection

        """
        self.SQLDBNAME = dbname if dbname else os.environ.get("RS_DBNAME")
        self.SQLHOST = host if host else os.environ.get("RS_HOST")
        self.SQLPORT = port if port else os.environ.get("RS_PORT")
        self.SQLUSER = user if user else os.environ.get("RS_USER")
        self.SQLPASSWORD = password if password else os.environ.get("RS_PASSWORD")

        self.CONN = psycopg2.connect(
            dbname=self.SQLDBNAME,
            host=self.SQLHOST,
            port=self.SQLPORT,
            user=self.SQLUSER,
            password=self.SQLPASSWORD,
        )

    def run_sql_redshift(self, query):
        """
        Pandas doesn't integrate with Redshift directly.
        Instead use psycopg2 to connect and transform results into a DataFrame manually.

        Parameters
        --------
        conn: cursor from database connection
        query: str

        Returns
        --------
        df: DataFrame

        """
        # Need to commit insert queries
        self.CONN.set_session(autocommit=True)

        with self.CONN.cursor() as cursor:
            # Execute query
            cursor.execute(query)

            # Add exceptions for queries that insert data only and don't return dataframes
            try:
                # Pull out column names from cursor
                colnames = [desc[0] for desc in cursor.description]

                # Fetch the entire query back
                data = cursor.fetchall()
            except Exception:
                pass

        try:
            # Transfer data to pandas dataframe
            df = pd.DataFrame(data, columns=colnames)

            return df

        # For queries that don't return data/write only queries
        except Exception:
            pass


class NotebookConnections:
    """
    Convert between Jupyter notebooks & Python scripts
    """

    def _remove_unused_lines(self, original_version):
        """
        Takes the content of a Jupyter notebook and removes code unneeded to produce final results.

        Parameters
        --------
        original_version: str

        Returns
        --------
        filtered_file: str
        """
        filtered_file = ""
        # Remove all the Jupyter notebook cell syntax
        for line in re.split("# In\[[0-9 ]*]:|# #", original_version):
            cleaned_line = line.replace("\n\n\n", "\n").strip()

            # If a cell contains multiple lines, but doesn't have any lines other than
            # code like comments or print statements
            if (
                "\n" in cleaned_line
                and len(
                    [
                        x
                        for x in cleaned_line.split("\n")
                        if x.strip() != ""
                        and x.strip()[0] != "#"
                        and x.strip()[:6] != "print("
                        and x.strip()[:4] != "for "
                        and x.strip() != "try:"
                        and x.strip()[:6] != "except"
                    ]
                )
                == 0
            ):
                print(f"Line skipped: {cleaned_line}")

            # Always include certain cells, such as those defining funcs
            elif (
                "def(" in cleaned_line
                or "\n" in cleaned_line
                or "inplace=True" in cleaned_line
                or "inplace = True" in cleaned_line
                or ("=" in cleaned_line.split("(")[0] and "(" in cleaned_line)
                or ("=" in cleaned_line and "(" not in cleaned_line)
            ):
                filtered_file += line.replace("\n\n\n", "\n")

            # Skip lines with common methods that aren't called with ``=`` assignment
            elif (
                cleaned_line == ""
                or "!pip install" in cleaned_line
                or cleaned_line == "#!/usr/bin/env python"
                or cleaned_line == "# coding: utf-8"
                or (cleaned_line[0] == "#")
                or ".head(" in cleaned_line
                or ".shape" in cleaned_line
                or "print(" in cleaned_line
                or "display(" in cleaned_line
                or ".apply" in cleaned_line
                or ".describe(" in cleaned_line
                or ".hist(" in cleaned_line
                or ".bar(" in cleaned_line
                or ".plot(" in cleaned_line
                or ".box(" in cleaned_line
                or ".area(" in cleaned_line
                or ".scatter(" in cleaned_line
                or ".kde(" in cleaned_line
                or ".loc" in cleaned_line
                or ".iloc" in cleaned_line
                or ".max(" in cleaned_line
                or ".min(" in cleaned_line
                or ".sum(" in cleaned_line
                or ".median(" in cleaned_line
                or ".info" in cleaned_line
                or "len(" in cleaned_line
                or "sns." in cleaned_line
                or (len(cleaned_line.split("\n")) == 1 and "(" not in cleaned_line)
                or (cleaned_line[0] == "[" and cleaned_line[-1] == "]")
            ):
                print(f"Line skipped: {cleaned_line}")

            else:
                filtered_file += line.replace("\n\n\n", "\n")

        return filtered_file

    def save_as_script(self, file_path):
        """
        Opens a Jupyter notebook file, cleans it, and saves as a Python script.

        Parameters
        --------
        file_path: str
        """
        os.system(f"jupyter nbconvert --to script {file_path}")

        script_file_path = file_path.replace(".ipynb", ".py")

        original_version = open(script_file_path).read()

        filtered_version = self._remove_unused_lines(original_version)

        with open(script_file_path, "w") as fh:
            fh.write(filtered_version)

        # Before saving as script, format and sort imports
        os.system(f"python -m black {script_file_path}")
        os.system(f"python -m isort {script_file_path}")

        print(f"Created Python script: {script_file_path}")

    def save_as_notebook(self, file_path):
        """
        Opens a Python script and saves as a Jupyter notebook.

        Parameters
        --------
        file_path: str
        """
        file_text = open(f"{file_path}").read()

        texts = []

        # Add cell definition for each new line group
        i = 1
        for text in file_text.split("\n\n#"):
            if i == 1:
                texts.append("\n# In[{}]: \n".format(str(i)) + text + "\n")
                i += 1
            elif text != "":
                texts.append("\n# In[{}]: \n#".format(str(i)) + text + "\n")
                i += 1

        temp_notebook_script = file_path.replace(".py", "_notebook.py")

        # First clean the text, then make a temp second script file
        with open(temp_notebook_script, "w") as fh:
            fh.write("".join(texts))

        # Convert temp cleaned script to notebook
        os.system(f"jupytext --to notebook {temp_notebook_script}")

        starting_notebook_name = temp_notebook_script.replace(".py", ".ipynb")
        updated_notebook_name = starting_notebook_name.replace("_notebook", "")

        # Clean up generated notebook name
        os.system(f"mv {starting_notebook_name} {updated_notebook_name}")

        # Remove temp intermediate script
        os.system(f"rm {temp_notebook_script}")

        print(f"Created Jupyter notebook: {updated_notebook_name}")


class SlackConnections:
    """
    Retrieve Slack messages
    """

    def __init__(self, slack_api_token=None):
        if not slack_api_token:
            slack_api_token = os.environ.get("SLACK_API_TOKEN")

        if not slack_api_token:
            return """
            No API key found. Follow these instructions to create one: https://slack.dev/python-slackclient/basic_usage.html.
            Then, either include as an argument or add as an environmental variable.
            """

        try:
            client = WebClient(token=slack_api_token)
        except Exception:
            return """
            Invalid API key or permissions.
            """

        self.client = client

    def _get_messages(
        self, time_filter, channel, remove_bot_messages, excluded_user_ids
    ):
        if remove_bot_messages:
            bot_filter = None
        else:
            bot_filter = "Include bots"

        response = self.client.conversations_history(
            channel=channel,
            scope="channels:history",
            latest=time_filter,
        )

        dict_data = response.data

        messages_list = [
            re.sub(
                # Remove channel names
                " -[A-Za-z0-9]*",
                " ",
                re.sub(
                    # Remove subteams
                    "!subteam\^[A-Za-z0-9]*",
                    " ",
                    re.sub(
                        # Removes channel numbers, links, emojis, usernames
                        "#[a-zA-Z-_]*|@[a-zA-Z]*|([a-zA-Z0-9_\.-]+)@([1-9a-zA-Z\.-]+)\.([a-zA-Z\.]{2,6})|<#[A-Z0-9 ]*\||:[a-zA-Z0-9-_]*:|<@[A-Z0-9]*>|(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
                        " ",
                        str(x["text"]).replace("|", " | "),
                    ),
                ),
            )
            .replace("\n", " ")
            .replace("<", " ")
            .replace(">", " ")
            .replace("|", " ")
            .encode("ascii", "ignore")
            .decode()
            .strip()
            for x in dict_data["messages"]
            if x.get("client_msg_id") is not bot_filter
            and x.get("user") not in excluded_user_ids
        ]

        try:
            time_filter = min([x["ts"] for x in dict_data["messages"]])
        except Exception:
            return messages_list, time_filter

        return messages_list, time_filter

    def get_messages(
        self,
        channels,
        remove_bot_messages=True,
        excluded_user_ids=[],
        messages_limit=np.inf,
    ):
        if not isinstance(channels, list):
            channels = [channels]

        all_messages = []

        for channel in channels:
            try:
                # First get latest messages and min timestamp
                messages_list, time_filter = self._get_messages(
                    "", channel, remove_bot_messages, excluded_user_ids
                )
            except Exception:
                return f"""
                Invalid API call. Make sure you have the right channel id, that your token includes the `channels:history` scope,
                and the bot has been added to the {channel} channel (by posting `@{{YOUR_BOTS_NAME}}` in the channel).
                """

            all_messages.extend(messages_list)

            # Then iterate until no more messages
            while channels:
                try:
                    messages_list, time_filter = self._get_messages(
                        time_filter, channel, remove_bot_messages, excluded_user_ids
                    )
                except Exception:
                    # Pausing for rate limits per minute
                    time.sleep(65)
                    messages_list, time_filter = self._get_messages(
                        time_filter, channel, remove_bot_messages, excluded_user_ids
                    )
                # Break when there aren't any more messages to fetch
                if messages_list == []:
                    break

                all_messages.extend(messages_list)

                if len(all_messages) >= messages_limit:
                    df = pd.DataFrame(all_messages, columns=["messages"])
                    df.drop_duplicates(inplace=True)
                    return df.head(messages_limit)

        df = pd.DataFrame(all_messages, columns=["messages"])
        df.drop_duplicates(inplace=True)

        return df
