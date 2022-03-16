from google.cloud import bigquery
from dotenv import load_dotenv
import os


def dataload():
    load_dotenv()
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/NLP_algorithm_auth_key/cryptonlp-333511-5df889e063ad.json"

    client = bigquery.Client()
    project = client.project
    dataset_ref = bigquery.DatasetReference(project, 'Crypixie')

    table_id = "cryptonlp-333511:Crypixie.article"

    schema = [
        # bigquery.SchemaField("id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("article_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
    ]


    table_ref = dataset_ref.table("article")
    table = bigquery.Table(table_ref, schema=schema)


    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json'
    dataset_id = 'Crypixie'
    table_id = 'article'
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    # job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    job_config = bigquery.LoadJobConfig(
        schema=[
            # bigquery.SchemaField("id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("article_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
        ],
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )


    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            location="europe-central2",  # Must match the destination dataset location.
            job_config=job_config,
        )

    # API request

    job.result()  # Waits for table load to complete.

    print("Loaded {} rows.".format(job.output_rows))
