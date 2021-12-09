from google.cloud import bigquery


# GOOGLE_APPLICATION_CREDENTIALS = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/NLP_algorithm_auth_key/cryptonlp-333511-5df889e063ad.json"
#
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/NLP_algorithm_auth_key/cryptonlp-333511-5df889e063ad.json"


def query_crypixie():
    client = bigquery.Client()
    query_job = client.query(
        """
        SELECT * 
FROM `cryptonlp-333511.Crypixie.article`
LIMIT 10"""
    )

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print("{} : {} views".format(row.url, row.view_count))


if __name__ == "__main__":
    query_crypixie()