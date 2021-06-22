import os
import pickle

import pandas as pd
import psycopg2
import s3fs


class S3Connections:
    def save_to_s3(
        self, directory_path, object_to_save, object_name,
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
        self.SQLDBNAME = dbname if dbname else os.environ.get("SQLDBNAME")
        self.SQLHOST = host if host else os.environ.get("SQLHOST")
        self.SQLPORT = port if port else os.environ.get("SQLPORT")
        self.SQLUSER = user if user else os.environ.get("SQLUSER")
        self.SQLPASSWORD = password if password else os.environ.get("SQLPASSWORD")

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
