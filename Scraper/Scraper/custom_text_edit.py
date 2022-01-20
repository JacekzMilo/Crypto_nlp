import json
from time import sleep


# class CustomText():
def customtextfunc(file):
    # file = open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json',
    #             "r")

    data = json.load(file)
    replaced_string = "".join(data['text'])

    data['text'] = replaced_string

    data = dict(data)

    dict1, dict2 = {}, {}

    dict1 = {"article_name": data["article_name"]}
    dict2 = {"text": data["text"]}



    # print(type(data))


    with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json", "w") as jsonFile:
        json.dump(dict1, jsonFile, ensure_ascii=False)

    with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.json", "w") as jsonFile:
        json.dump(dict2, jsonFile, ensure_ascii=False)

    # sleep(5)
    with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json", "r+") as jsonFile:
        jsonFile.truncate(0)
        # json.dump(jsonFile, ensure_ascii=False)

    f1data = f2data = ""

    with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json') as f1:
        f1data = f1.read()

    with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.json') as f2:
        f2data = f2.read()

    f1data += "\n"
    f1data += f2data

    with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json', 'a') as f3:
        f3.write(f1data)