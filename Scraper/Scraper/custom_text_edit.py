import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from googletrans import Translator
from Crypto_nlp.Load_to_GCP import load
from time import sleep


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


##########################
#Po scrapowaniu kazde zdanie jest z apostrofami, jest w postaci słownika gdzie klucze to: article_name i article_text, wartość dla article_text jest w postaci listy
#Outputem funkcji customtextfunc jest plik .csv ktory jest w postaci słownika tylko z zewnętrznymi apostrofami
#Outputem article_manipulation jest przetłumaczony tekst

def customtextfunc(file):
    # file = open(f'{file}', "r")
    with open(f'{file}') as f:
        data = f.read()
    df = data.replace('][', ',')
    data = pd.read_json(df)
    print(data['article_text_1'].dropna())
    # data = json.load(file)
    # data=pd.read_csv(f'{file}', encoding="utf8")
    # print("data", data)
    j=0
    for key in data:
        # print('key', key)
        for i in range(1, 2):
            if key == f"article_text_{i}":
                # data[f"article_text_{i}"].drop()
                data_df=pd.DataFrame(data)
                data_df[f"article_text_{i}"] = data_df[f"article_text_{i}"].dropna()
                print('data_df[f"article_text_{i}"]', data_df[f"article_text_{i}"])

                # data[f"article_text_{i}"]=data[f"article_text_{i}"].to_string()
                data_string = data[f"article_text_{i}"].to_string()
                # print("data_string", type(data_string))
                replaced_string = data_string.strip('"')
                # replaced_string = " ".join(data_string)
                data[f"article_text_{i}"] = replaced_string

                print(f'data["article_text_{i}"]',data[f"article_text_{i}"])
                # print(f'data["article_text_{i}"]', type(data[f"article_text_{i}"]))

                data = dict(data)

                dict1, dict2 = {}, {}

                dict1 = {"article_name": data[f"article_name_{i}"]}
                dict2 = {"article_text": data[f"article_text_{i}"]}
                print("dic1", dict1)
                print("dic2", dict2)

                with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json", "w") as jsonFile:
                    json.dump(dict1, jsonFile, ensure_ascii=False)

                with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.json", "w") as jsonFile:
                    json.dump(dict2, jsonFile, ensure_ascii=False)

                # sleep(5)
                with open("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json", "r+") as jsonFile:
                    jsonFile.truncate(0)

                f1data = f2data = ""

                with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json') as f1:
                    f1data = f1.read()

                with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.json') as f2:
                    f2data = f2.read()

                f1data += "\n"
                f1data += f2data

                if j==0:
                    with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json', 'a') as f3:
                        f3.write(f1data)
                else:
                    with open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json', 'a') as f3:
                        data = json.load(f3)
                        data.update(f1data)
                j += 1

    txt = (
        "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json")
    records = map(json.loads, open(txt, encoding="utf8"))
    df = pd.DataFrame.from_records(records)
    df["ommit"]=range(len(df))
    df.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv',
        index=False, header=True)

    sleep(1)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv'
    # load(filename, 'article')

def article_manipulation(file):
    #     df = pd.read_csv(f"{file}")
    # # print("file", file)
    # # records = map(json.loads, open(file, encoding="utf8"))
    # # df = pd.DataFrame(file)
    df=pd.read_csv("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv", encoding="utf8")
    # print("df", df)
    # for key in file:
    #     for i in range(1, 2):
    #         if key==f"article_text_{i}":
    text = df["article_text"].loc[1:]
    text_str = text.to_string(index=False)
    # print('text_str1', text_str)

    text_headline=df['article_name'].loc[:0,]
    text_headline=text_headline.to_string(index=False)
    # print('text_headline1',text_headline)


    def unique_list(l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist

    # text_str=' '.join(unique_list(text_str.split()))
    # # print('text_str2', text_str)
    #
    # text_headline=' '.join(unique_list(text_headline.split()))
    # # print('text_headline2',text_headline)

    #######################Tokenizacja tekstu i usuwanie integerów
    # tokenizer = RegexpTokenizer(r"\w+")
    # text_str= tokenizer.tokenize(text_str)
    # text_str = [x for x in text_str if not (x.isdigit()
    #                                          or x[0] == '-' and x[1:].isdigit())]
    # text_headline= tokenizer.tokenize(text_headline)
    # text_headline = [x for x in text_headline if not (x.isdigit()
    #                                          or x[0] == '-' and x[1:].isdigit())]
    #######################

    ####################### Zamiana tekstu na string
    # def listToString(text_str):
    #     # initialize an empty string
    #     str1 = " "
    #     return (str1.join(text_str))
    # text_str = listToString(text_str)
    # text_headline = listToString(text_headline)
    # print('text_str', text_str)
    # print('text_headline', text_headline)

    #######################

    ####################### tu jest tłumaczenie
    translator = Translator()

    text_str_translated = translator.translate(text_str, src='auto', dest='en')
    text_str= text_str_translated.text
    # print('text_str3', text_str)

    text_headline_translated=translator.translate(text_headline, src='auto', dest='en')
    text_headline=text_headline_translated.text
    # print('text_headline3', text_headline)

    text_str_dic={"article_name":[text_headline],"article_text_translated":[text_str]}
    article_translated=pd.DataFrame.from_dict(text_str_dic)

    article_translated["ommit"]=range(len(article_translated))

    # print('article_translated', article_translated)
    article_translated.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv', index = False, header=True)
    sleep(1)
    # filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv'
    # load(filename, 'article_translated')
    print("Oryginalny tekst przetłumaczony i zapisany jako article_translated.csv")