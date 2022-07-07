import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from googletrans import Translator
from Crypto_nlp.Load_to_GCP import load
from time import sleep
from io import StringIO

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


######################
#After scraping, each sentence has single quotes because they are in separate <p> positions, it is in .json file in the form of a dictionary where
#keys are article_name and the value is the article_text, the value for article_text is as a list
#Output of the customtextfunc function is a .csv file which is in the form of a dictionary with only outer quotes
#The article_manipulation output is a translated text
######################


###################### Below function opens the .json file, changes text in lists into one strig and saves it in Scraped_data_no_quotes2.csv file.
# Does the same thing for article title, assigns id's to articles and appends links
def customtextfunc(file):
    # file = open(f'{file}', "r")
    with open(f'{file}', 'r', encoding='utf8') as f:
        data = f.read()
    df = data.replace('][', ',')
    data = pd.read_json(StringIO(df))
    print()

    # Function to convert list to string
    def listToString(s):
        # initialize an empty string
        str1 = " "
        return (str1.join(s))

    j=0
    for i in data['article_text']:
        data_string = listToString(i)
        replaced_string = " ".join(data_string.split())
        data_df =pd.DataFrame(data=[replaced_string] ,columns=["article_text"])
        if j == 0:
            article_text_df = data_df
        else:
            article_text_df = article_text_df.append(data_df)
            article_text_df.to_csv(
                r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.csv',
                index=False, header=True)
        j += 1


    j = 0
    for i in data["article_name"]:
        article_name_df = pd.DataFrame(data=[i], columns=["article_name"])
        if j == 0:
            article_name_df1 = article_name_df
        else:
            article_name_df1 = article_name_df1.append(article_name_df)
        j+=1


    article_name_df1.reset_index(inplace=True)

    article_link = pd.DataFrame(data['article_link'])

    article_name_df1 = pd.concat([article_link, article_name_df1], axis=1).drop(columns=['index'])
    article_name_df1['id'] = int()

    #ID generator - for now not used
    def count_non_digits(s):
        count = 0
        for i in range(len(s)):
            if not s[i].isdigit():
                count = count + 1

        return count

    # assigning Id's
    ids = [1049, 1108, 1026, 1077, 1069, 1055]
    i=0

    for c in ids:
        article_name_df1['id'][i:] = c
        i += 1

    # i=0
    # for c in article_name_df1['article_name']:
    #     article_name_df1['id'][i:] = count_non_digits(c)+1000
        # print("second",count_non_digits(c)+1000)
        # i+=1
    # print(article_name_df1['id'])

    first_col = article_name_df1.pop("id")
    article_name_df1.insert(0, 'id', first_col)
    ##########################


    ########################## Below code joins two .csv files into one
    print("article_name_df1", article_name_df1)

    article_name_df1.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.csv',
        index=False, header=True)

    article_name_df = pd.read_csv('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.csv', encoding="utf8")

    article_text_df = pd.read_csv('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes2.csv', encoding="utf8")

    df_total = pd.concat([article_name_df, article_text_df], axis=1)

    #Here empty Scraped_data.csv file is created
    filename = "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv"
    # opening the file with w+ mode truncates the file
    f = open(filename, "w+")
    f.close()

    df_total.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv',
        index=False, header=True)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv'
    load(filename, 'article')
    print("Oryginalny tekst wyczyszczony, zapisany jako Scraped_data.csv i wrzucoy do tabeli article" )
##########################


########################## Below code translates original text to english
def article_translation(file):

    df=pd.read_csv("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv", encoding="utf8")

    ########################## translation function
    def translate(string):
        translator = Translator()
        text_str_translated = translator.translate(string, src='auto', dest='en')
        return text_str_translated.text

    j = 0
    for i in df["article_text"]:
        text = df["article_text"].loc[j]
        text_translated = translate(text)
        text_translated_df = pd.DataFrame(data=[text_translated], columns=["article_text"])

        if j == 0:
            text_translated_df1 = text_translated_df

        else:
            text_translated_df1 = pd.concat([text_translated_df1, text_translated_df], axis=0)

        j+=1

    j = 0
    for i in df["article_name"]:
        text = df["article_name"].loc[j]
        article_name_translated = translate(text)
        article_name_translated_df = pd.DataFrame(data=[article_name_translated], columns=["article_name"])

        if j == 0:
            article_name_translated_df1 = article_name_translated_df

        else:
            article_name_translated_df1 = pd.concat([article_name_translated_df1, article_name_translated_df], axis=0)
        j += 1
    ##########################


    ########################## Below code puts translated text, article links and Id's into article_translated.csv and then
    # loads it into BQ
    article_link = pd.DataFrame(df['article_link'])
    article_name_translated_df1.reset_index(inplace=True)
    text_translated_df1.reset_index(inplace=True)
    id_df = pd.DataFrame(df['id'])

    text_translated_all = pd.concat([id_df, article_name_translated_df1, text_translated_df1, article_link], axis=1).drop(columns=['index'])
    print('text_translated_all', text_translated_all)

    text_translated_all.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv', index = False, header=True)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv'
    load(filename, 'article_translated')
    print("Oryginalny tekst przetłumaczony i zapisany jako article_translated.csv")
    ##########################