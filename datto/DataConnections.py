import json
import os
import pickle
from datetime import datetime
from time import sleep, time
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
import psycopg2
import requests
import s3fs
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from tqdm import tqdm

from datto.Setup import Setup

load_dotenv()


class DataConnections:
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

    def setup_redshift_connection(
        self, dbname=None, host=None, port=None, user=None, password=None
    ):
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

        conn = psycopg2.connect(
            dbname=self.SQLDBNAME,
            host=self.SQLHOST,
            port=self.SQLPORT,
            user=self.SQLUSER,
            password=self.SQLPASSWORD,
        )
        return conn

    def run_sql_redshift(self, conn, query):
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
        with conn.cursor() as cursor:
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


class KafkaInterface:
    def __init__(
        self,
        topic,
        bootstrap_servers=None,
        rest_url=None,
        kafka_manager=None,
        rest_auth=None,
        dummy=None,
        is_rest=False,
    ):
        """Pulls credentials from environment automatically if set."""
        self.TOPIC_NAME = topic
        self.BOOTSTRAP_SERVERS = (
            bootstrap_servers
            if bootstrap_servers
            else os.environ.get("BOOTSTRAP_SERVERS")
        )
        self.KAFKA_REST_URL = rest_url if rest_url else os.environ.get("KAFKA_REST_URL")
        self.KAFKA_MANAGER = (
            kafka_manager if kafka_manager else os.environ.get("KAFKA_MANAGER")
        )
        self.KAFKA_REST_AUTH = (
            rest_auth if rest_auth else os.environ.get("KAFKA_REST_AUTH")
        )
        self.DUMMY_KAFKA = dummy
        self.is_rest = is_rest
        self.producer = None

        log = Setup()
        self.logger = log.setup_logger()

        if not self.is_rest:
            # This will raise if it cannot get any brokers, probably better to fail here then at send time
            dc = DataConnections()
            self.producer = self.get_kafka_producer()

        self.logger.info(
            f"Kafka producer is {bool(self.producer)} for {self.TOPIC_NAME}, is_rest -> {self.is_rest}"
        )

        # Get connection info
        try:
            self.KAFKA_AUTH_SPLIT = tuple(self.KAFKA_REST_AUTH.split(":"))
        except KeyError:
            self.logger.critical("Missing KAFKA_REST_AUTH Env Var")
        except Exception as e:
            self.logger.critical(
                f"Unhandled error when importing KAFKA_REST_AUTH env var: {e}"
            )

    def get_mandatory_fields(self):
        """Get all fields that are mandatory for a Kafka message coming from this tool"""
        ts = time()
        return {
            "timestamp": ts,
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "_partition_key": "date",
            "message_bundle_id": str(uuid4()),
        }

    def set_defaults(self, lod):
        """Package up list of dicts, by setting defaults"""
        mandatory_fields = self.get_mandatory_fields()
        for k, v in mandatory_fields.items():
            # We use .setdefault rather than .update to avoid overwriting values that have been set by the sender
            [i.setdefault(k, v) for i in lod]
        return lod

    def send(self, lod):
        """"Send a list of dictionaries to the Kafka REST API """
        lod_plus_defaults = self.set_defaults(lod)
        if self.is_rest:
            response = requests.post(
                url=f"{self.KAFKA_REST_URL}/topics/{self.TOPIC_NAME}",
                json={"records": [{"value": i} for i in lod_plus_defaults]},
                auth=self.KAFKA_AUTH_SPLIT,
                headers={"Content-Type": "application/vnd.kafka.json.v2+json"},
            )
            try:
                response.raise_for_status()
            except requests.exceptions.Timeout:
                self.logger.warning("Timeout on post to Kafka API")
                # Maybe set up for a retry, or continue in a retry loop
            except requests.exceptions.TooManyRedirects:
                # Tell the user their URL was bad and try a different one
                self.logger.warning("Too many redirects on post to Kafka API")
            except requests.exceptions.HTTPError as e:
                self.logger.warning(
                    f"Got a {e.response.status_code} because {e.response.reason}"
                )
            except requests.exceptions.RequestException as e:
                self.logger.info(
                    f"We're eating this message for some reason... it seemed like a good idea at the time: {e}"
                )
            return response
        else:
            self.publish_to_kafka(lod_plus_defaults)

    def publish_to_kafka(self, data: List[Dict[str, Any]]):
        if self.DUMMY_KAFKA is not None:
            self.logger.warn("DUMMY_KAFKA is set")
            self.logger.warn(f"self.logger.infoing data")
            for obj in data:
                self.logger.info(obj)
            return

        n_records = len(data)
        if n_records > 0:
            self.logger.info(
                f"started_publishing to {self.TOPIC_NAME}, records: {n_records}"
            )
            for record in tqdm(data):
                self.producer.send(self.TOPIC_NAME, record)
            self.producer.flush()
            metrics = self.producer.metrics()
            self.logger.debug(metrics)
            self.logger.info(
                f"finished_publishing to {self.TOPIC_NAME}, records: {n_records}"
            )
            try:
                self.logger.info(
                    f"{self.TOPIC_NAME} producer metrics: {json.dumps(metrics)}"
                )
            except OverflowError:
                # Sometimes we hit an overflow error preserving the log so we can handle it better
                self.logger.info(f"{self.TOPIC_NAME} producer metrics: {str(metrics)}")

    def get_kafka_producer(self):
        no_brokers_retry = 0
        while True:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode(),
                    acks=1,
                    retries=5,
                )
                return producer
            except NoBrokersAvailable as e:
                no_brokers_retry += 1
                sleep(1)
                if no_brokers_retry > 5:
                    raise e
