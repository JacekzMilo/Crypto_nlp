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
#Po scrapowaniu kazde zdanie jest z apostrofami ponieważ sa w oddzielnych pozycjach <p> , jest w postaci słownika gdzie
#klucze to: article_name i article_text, wartość dla article_text jest w postaci listy
#Outputem funkcji customtextfunc jest plik .csv ktory jest w postaci słownika tylko z zewnętrznymi apostrofami
#Outputem article_manipulation jest przetłumaczony tekst


def customtextfunc(file):
    # file = open(f'{file}', "r")
    with open(f'{file}', 'r', encoding='utf8') as f:
        data = f.read()
    df = data.replace('][', ',')
    data = pd.read_json(df)

    dict1, dict2 = {}, {}

    j=0

    for i in data['article_text']:

        # Function to convert list to string
        def listToString(s):
            # initialize an empty string
            str1 = " "
            return (str1.join(s))

        data_string = listToString(i)

        replaced_string = " ".join(data_string.split())

        data_df =pd.DataFrame(data=[replaced_string] ,columns=["article_text"])


        dict1 = {"article_name": data["article_name"][0]}

        df1 = pd.DataFrame(dict1, index=[0])
        # print("df1_przed_petlami", df1)
        if j == 0:
            article_text_df = data_df
            # print("article_text_j=0", article_text_df)
            df1.to_csv(
                r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.csv',
                index=False, header=True)

            data_df.to_csv(
                r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.csv',
                index=False, header=True)
        else:
            article_name_df=pd.DataFrame(data=[data["article_name"][j]], columns=["article_name"])
            # print("article_df", article_df)
            df1 = df1.append(article_name_df)
            # print("df1", df1)
            df1.to_csv(
                r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.csv',
                index=False, header=True)
            # print('druga runda')
            # print("data_df", data_df)

            article_text_df = article_text_df.append(data_df)
            # print("article_text_df", article_text_df)
            article_text_df.to_csv(
                r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.csv',
                index=False, header=True)
        j += 1

    article_name_df = pd.read_csv('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.csv', encoding="utf8")

    # print('article_name_df', article_name_df)

    article_text_df = pd.read_csv('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.csv', encoding="utf8")


    df_total = pd.concat([article_name_df, article_text_df], axis=1)
    # print('df_total', df_total)


    #     #to tworzy pusty Scraped_data.csv
    filename = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv"
    # opening the file with w+ mode truncates the file
    f = open(filename, "w+")
    f.close()

    #to jest potrzebne zeby BQ rozkminił nagłowki tabel, do pominięcia przy wykresach
    df_total["ommit"]=range(len(df_total))
    # print("df_total", df_total)
    df_total.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv',
        index=False, header=True)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv'
    load(filename, 'article')

def article_translation(file):

    df=pd.read_csv("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv", encoding="utf8")
    # print("df", df)
    # for key in file:
    #     for i in range(1, 2):
    #         if key==f"article_text_{i}":

    ####################### tu jest tłumaczenie
    def translate(string):
        translator = Translator()
        text_str_translated = translator.translate(string, src='auto', dest='en')
        return text_str_translated.text


    j=0
    for i in df["article_text"]:
        text = df["article_text"].loc[j]
        text_translated = translate(text)
        # print('text_translated', text_translated)
        text_translated_df = pd.DataFrame(data=[text_translated], columns=["article_text"])

        if j == 0:
            text_translated_df1 = pd.DataFrame(data=[text_translated], columns=["article_text"])
            # print('text_translated_df1', text_translated_df1)

        else:
            text_translated_df2 = pd.concat([text_translated_df1, text_translated_df], axis=0)
            # print("drugra runda")
            print('text_translated_df2', text_translated_df2)
        j+=1

    j = 0
    for i in df["article_name"]:
        text = df["article_name"].loc[j]
        text_translated = translate(text)
        # print('text_translated', text_translated)
        text_translated_df = pd.DataFrame(data=[text_translated], columns=["article_name"])

        if j == 0:
            text_translated_df1 = pd.DataFrame(data=[text_translated], columns=["article_name"])
            # print('text_translated_df1', text_translated_df1)

        else:
            text_translated_df3 = pd.concat([text_translated_df1, text_translated_df], axis=0)
            # print("drugra runda")
            # print('text_translated_df3', text_translated_df3)
        j += 1

    text_translated_all = pd.concat([text_translated_df3, text_translated_df2], axis=1)
    print('text_translated_all', text_translated_all)

    text_translated_all["ommit"]=range(len(text_translated_all))

    text_translated_all.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv', index = False, header=True)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv'
    load(filename, 'article_translated')
    print("Oryginalny tekst przetłumaczony i zapisany jako article_translated.csv")