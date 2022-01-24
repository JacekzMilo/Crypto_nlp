from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import urllib.request
import nltk
import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from google.cloud import bigquery
from dotenv import load_dotenv
import os
# response = urllib.request.urlopen('https://en.wikipedia.org/wiki/SpaceX')
# html = response.read()
# soup = BeautifulSoup(html,'html5lib')

pd.options.display.max_colwidth = 5000000
file = (
    "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng.json")
records = map(json.loads, open(file, encoding="utf8"))
df = pd.DataFrame.from_records(records)

text = df.loc[1:, ["text"]]
text_str = text.to_string()
text_str= text_str.lower()

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

text_str=' '.join(unique_list(text_str.split()))
print(text_str)
tokenizer = RegexpTokenizer(r"\w+")
text_str= tokenizer.tokenize(text_str)

# print(text_str)

text_str = [x for x in text_str if not (x.isdigit()
                                         or x[0] == '-' and x[1:].isdigit())]
# print(no_integers)
def listToString(text_str):
    # initialize an empty string
    str1 = " "
    return (str1.join(text_str))


text_str = listToString(text_str)

# print(text_str)

text_strip = text_str.strip()

tokens = [t for t in text_strip.split()]

sr = stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)
freq = nltk.FreqDist(clean_tokens)



dic = {}
keys = []
values = []
for key, val in freq.items():
    keys.append('{}'.format(key))
    values.append('{}'.format(val))

# print(keys)
dic = dict(zip(keys, values))
print(dic)

freq.plot(20, cumulative=False)


# with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_refactored_text.json", "w") as jsonFile:
#     json.dump(dic, jsonFile)

# load_dotenv()
# os.environ[
#     "GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/NLP_algorithm_auth_key/cryptonlp-333511-5df889e063ad.json"
#
# client = bigquery.Client()
# project = client.project
# dataset_ref = bigquery.DatasetReference(project, 'Crypixie')
#
# table_id = "cryptonlp-333511:Crypixie.article_transformed"
#
# # schema = [
# #     # bigquery.SchemaField("id", "STRING", mode="NULLABLE"),
# #     bigquery.SchemaField("article_name", "STRING", mode="NULLABLE"),
# #     bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
# # ]
#
#
# table_ref = dataset_ref.table("article_transformed")
# table = bigquery.Table(table_ref)
#
#
# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_refactored_text.json'
# dataset_id = 'Crypixie'
# table_id = 'article_transformed'
# dataset_ref = client.dataset(dataset_id)
# table_ref = dataset_ref.table(table_id)
# # job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
# job_config = bigquery.LoadJobConfig(
#     # schema=[
#     #     # bigquery.SchemaField("id", "STRING", mode="NULLABLE"),
#     #     bigquery.SchemaField("article_name", "STRING", mode="NULLABLE"),
#     #     bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
#     # ],
#     source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
#     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
#     autodetect=True
# )
#
#
# with open(filename, "rb") as source_file:
#     job = client.load_table_from_file(
#         source_file,
#         table_ref,
#         location="europe-central2",  # Must match the destination dataset location.
#         job_config=job_config,
#     )  # API request
#
# job.result()  # Waits for table load to complete.
#
# print("Loaded {} rows.".format(job.output_rows))