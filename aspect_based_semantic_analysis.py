def nlp_article_semantic(file):
    from Crypto_nlp.Load_to_GCP import load
    from textblob import TextBlob, Word
    import seaborn as sns
    sns.set(color_codes=True)
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from string import punctuation
    import re
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    from nltk.corpus import stopwords

    # txt = (
    #     f"{file}")
    # records = map(json.loads, open(txt, encoding="utf8"))
    # df = pd.DataFrame.from_records(records)
    # result=df["text"]
    df = pd.read_csv(f"{file}", encoding="utf8")

    result = df["article_text"]


    ###################### Below is a func that cleans passed text
    def clean_sentence(sentence):
        sentence = re.sub(r"(?:\@|https?\://)\S+|\n+", "", sentence.lower())
        # Fix spelling errors in comments!
        sent = TextBlob(sentence)
        sent.correct()
        clean = ""
        for sentence in sent.sentences:
            words = sentence.words
            # Remove punctuations
            words = [''.join(c for c in s if c not in punctuation) for s in words]
            words = [s for s in words if s]
            clean += " ".join(words)
            clean += ". "
        return clean
    ######################


    ###################### Below code cleans article text and adds it to a list separated by article
    result_after_clean=[]
    j=0
    for i in result:
        if j == 0:
            result_after_clean = (pd.Series(clean_sentence(i)))
        else:
            result_after_clean2 = (pd.Series(clean_sentence(i)))
            result_after_clean = pd.concat([result_after_clean, result_after_clean2], axis=0)
        j+=1

    print('result_after_clean2', result_after_clean)
    result=result_after_clean

    ###################### Below code checks article sentiment polarity, used in polarity distribution plot
    sentiment_scores = list()
    i = 0
    for article in result:
        line = TextBlob(article)
        sentiment_scores.append(line.sentiment.polarity)
        if(i <= 10000):
            print(article + ": POLARITY=" + str(line.sentiment.polarity))
            i += 1
    ######################
    # Convert array of comments into a single string

    featureList=['litecoin', 'polkadot', 'bitcoin', 'stellar', 'dogecoin', 'binance', 'tether', 'monero', 'solana',
             'avalanche', 'chainlink', 'algorand', 'polygon', 'vechain', 'tron', 'zcash', 'eos', 'tezos', 'neo',
             'stacks', 'nem', 'decred', 'storj', '0x', 'digibyte']
    ######################


    ###################### Below code check how many times each key word appears in the article
    comments = TextBlob(' '.join(result))
    feature_count = dict()
    for phrase in featureList:
        count = 0
        for word in phrase.split():
            if word not in stopwords.words('english'):
                count += comments.words.count(word)

        print(phrase + ": " + str(count))
        feature_count[phrase] = count

    threshold = len(featureList)/100

    print("Threshold:" + str(threshold))

    frequent_features = list()

    # here code checks which  coin appears the most often
    for feature, count in feature_count.items():
        if count >= threshold:
            frequent_features.append(feature)
    print(' Features:')
    print(frequent_features)

    # Here is sentence sentiment polarity analyzer
    def nltk_sentiment(sentence):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk_sentiment = SentimentIntensityAnalyzer()
        score = nltk_sentiment.polarity_scores(sentence)
        return score
    ######################


    ###################### Below code adds ID's to every article
    comment_id = []
    i=0
    for comment in result:
        comment_ser = [[comment]]
        comment_ser.append([df['id'][i]])
        comment_id.append(comment_ser)
        i += 1
    ######################


    ###################### Below code searches sentences with key words. Creates a dictionary where keys are the coin names and values are sentences
    # and a list of sentences with key words and article ID that they came from

    sent_list = []
    absa_list = dict()

    for f in frequent_features:
        absa_list[f] = list()

    j = 0
    for i in comment_id:
        for articl in i[0]:
            blob = TextBlob(articl)
            art_sent = []
            for f in frequent_features:
                q = '|'.join(f.split())
                # For each key word
                # For each sentence of the comment
                for sentence in blob.sentences:
                    # Search for frequent feature 'f'
                    if re.search(r'\w*(' + str(q) + ')\w*', str(sentence)):
                        # print("f", f)
                        # print("sentence", sentence)
                        absa_list[f].append(sentence)
                        # print("absa_list[f]", absa_list)
                        art_sent.append([sentence])

            if art_sent:

                art_sent.append(comment_id[j][1])
                sent_list.append(art_sent)
                j += 1

            else:
                j += 1

    print('sent_list', sent_list) # A list of sentences with key words and article ID that they came from
    print("absa_list", absa_list) # A dictionary where keys are the coin names and values are sentences
    print("absa_list.values", absa_list.values())
    ######################


    ###################### Creates a list of article ID's that contains key words
    id_list=[]
    sentences=[]

    for i in sent_list:
        for j in i[: -1]:
            sentences.append(" ".join(str(x) for x in j))
        id_list.append([x for x in i[-1]])

    print('id_list', id_list) # ID list of sentences that contain key words
    ######################


    ###################### Below code multiplies number of ID so it could match dataframe created in next step
    id_df_new = pd.DataFrame()
    j=0
    for i in sent_list:
        id_df = pd.DataFrame(id_list)
        id_df = pd.DataFrame([id_df.iloc[j]]*(len(i)-1))
        j += 1
        id_df_new = pd.concat([id_df_new, id_df], ignore_index=True)
    ######################


    ###################### Below code creates dataframe with article ID, sentences and their sentiment analysis. Loads it to results_for_plot
    nltk_results = [nltk_sentiment(row) for row in sentences]
    results_df = pd.DataFrame(nltk_results)
    text_df = pd.DataFrame(sentences)
    nltk_df2 = pd.DataFrame()
    nltk_df2['id'] = id_df_new
    nltk_df2['text'] = text_df
    nltk_df2 = nltk_df2.join(results_df)
    nltk_df2['neg'] = nltk_df2['neg']*(-1)
    print('nltk_df2', nltk_df2)

    nltk_df2.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv',
        index=False, header=True)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv'
    load(filename, 'results_for_plot')
    print("Przetłumaczony tekst przeanalizowany, zapisany jako article_semantics_results_for_plot.csv i wrzucony do tabeli results_for_plot")
    ######################


    ###################### Below code performs sentence sentiment polarity and creates two dictionaries
    scores = list()
    absa_scores = dict()
    for k, v in absa_list.items():
        absa_scores[k] = list()
        id_all = list()
        for sent in v:
            score = sent.sentiment.polarity # This is fo a sentence
            scores.append(score)
            absa_scores[k].append(score)

            for sent_nltk in nltk_df2['text']:
                row_id = nltk_df2[nltk_df2['text'] == sent_nltk].index[0]
                if re.search(r'\w*(' + str(sent) + ')\w*', str(sent_nltk)):
                    id = nltk_df2['id'][row_id]
                    id_all.append(id)

        absa_scores[k].append(id_all)

    print("absa_list2", absa_list) # dictionary with coin name as a key and sentences as a value
    print("absa_scores", absa_scores) # dictionary with coin name as a key and sentence sentiment polarity as a value
    ######################


    ###################### Below code creates two dictionaries: sem_dic with sentiment analysis column and coin names column and coin_dic with
    # sentences in one column and coin names in second column.
    sent_all=[]
    coin_list=[]
    sentence_sentiment_list=[]
    final_list=[]
    coin_dic = {}
    sem_dic = {}

    i=0
    for sent in sentences:
        for coin in frequent_features:
            if coin in sent:
                sentence_sentiment_list.append(nltk_results[i])
                coin_list.append(coin)
                sem_dic = {"Sentence sentiment analysis": sentence_sentiment_list, "Coin": coin_list}

                final_list.append(coin)
                sent_all.append(sent)
                coin_dic = {"Coin": coin_list, "Sentence": sent_all}
                i+=1

    print("sem_dic", sem_dic) # Dictionary with sentiment analysis column and coin names column
    print("coin_dic", coin_dic) # Dictionary with sentences in one column and coin names in second column
    ######################


    ###################### Below code grupps sentences and their sentiment analysis by coin names and puts it in final_df_lists dataframe.
    # Then it loads this dataframe to results_aggregaded
    final_df_1=pd.DataFrame.from_dict(coin_dic)
    final_df_1_grupped = final_df_1.groupby('Coin')
    final_df_lists1 = final_df_1_grupped['Sentence'].apply(list).reset_index()

    final_df_2=pd.DataFrame.from_dict(sem_dic)
    final_df_2_grupped = final_df_2.groupby('Coin')
    final_df_lists2 = final_df_2_grupped['Sentence sentiment analysis'].apply(list).reset_index().drop(columns=["Coin"])

    final_df_lists3 = pd.DataFrame(final_df_lists2)

    final_df_lists = pd.concat([final_df_lists1, final_df_lists3],axis=1)


    final_df_lists["ommit"]=range(len(final_df_lists)) #to jest tylko po to zeby BQ rozpoznawal nazwy kolumn
    print("final_df_lists", final_df_lists)

    final_df_lists.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_aggregaded.csv',
        index=False, header=True)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_aggregaded.csv'
    load(filename, 'results_aggregaded')
    print("Przetłumaczony tekst przeanalizowany, zapisany jako article_semantics_results_aggregaded.csv i wrzucony do tabeli results_aggregaded")
    ######################


    # ################################ Bar plot
    # #plot1
    # n_groups = len(frequent_features)
    # positive = nltk_df2['pos'].head(len(frequent_features))
    # negative = nltk_df2['neg'].head(len(frequent_features))
    #
    # # create plot
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.3
    # opacity = 1
    #
    # rects1 = plt.bar(index, positive, bar_width,
    #                  alpha=opacity,
    #                  color='b',
    #                  label='positive sentiments')
    #
    # rects2 = plt.bar(index + bar_width, negative, bar_width,
    #                  alpha=opacity,
    #                  color='r',
    #                  label='negative sentiments')
    #
    # plt.xlabel('Features')
    # plt.ylabel('sentiment value')
    # plt.title('Top features and its sentiment')
    # plt.xticks(index + bar_width, nltk_df2['text'])
    # plt.legend()
    # fig.set_size_inches(15, 10)
    # plt.show()
    # ##########################
    #
    #
    # ######################### Polarity distribution plot -> two bar plots
    # #Plot 2
    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
    # plot1 = sns.distplot(scores, ax=ax1)
    #
    # ax1.set_title('Aspect wise scores')
    # ax1.set_xlabel('Sentiment Polarity')
    # ax1.set_ylabel('# of comments')
    #
    # ax2.set_title('Article wise scores')
    # ax2.set_xlabel('Sentiment Polarity')
    # ax2.set_ylabel('# of comments')
    #
    # plot2 = sns.distplot(sentiment_scores, ax=ax2)
    # plt.show()
    # ##########################


    ########################## Create data values for stripplot and boxplot
    vals = dict()
    vals["aspects"] = list()
    vals["scores"] = list()
    vals["id"] = list()

    for k, v in absa_scores.items():
        for score in v:
            if type(score) is float:
                vals["aspects"].append(k)
                vals["scores"].append(score)
            if type(score) is list:
                for i in score:
                    vals["id"].append("".join(str(i)))

    # vals["id"] = ", ".join(str(x) for x in vals["id"])
    print("vals", vals)
    print("vals[scores]", vals["scores"])
    vals_df = pd.DataFrame.from_dict(vals)
    vals_df.to_csv(
            r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/box_plot.csv',
            index=False, header=True)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/box_plot.csv'
    load(filename, 'box_plot')
    ##########################


    # # ######################### Polarity distribution plot -> box plot
    # # #Plot 3
    #
    # fig, ax1 = plt.subplots(figsize=(15, 10))
    # color = sns.color_palette("Blues")
    # plt.xticks(rotation=90)
    # sns.set_context("paper", font_scale=3)
    # sns.boxplot(x="aspects", y="scores", data=vals, palette=color, ax=ax1)
    #
    # #ten chyba najmniej istotny, inne pokazanie wykresu 1
    # color = sns.color_palette("Blues", 1)
    # fig, ax1 = plt.subplots(figsize=(15, 10))
    # plt.xticks(rotation=90)
    # sns.set_context("paper", font_scale=5)
    # sns.stripplot(y="aspects", x="scores", data=vals, palette=color)
    # plt.show()
    # ##########################


    ######################## Sentences grupped by article ID, might be usefull
    nltk_df2_grupped1 = nltk_df2.groupby('id')
    nltk_df2_grupped = nltk_df2_grupped1['text'].apply(list).reset_index()
    print('nltk_df2_grupped', nltk_df2_grupped)
    print('final_df_lists1', final_df_lists1)

    i=0
    sentence_id = pd.DataFrame(columns = ['id', 'sentence'])
    for sent_grupped in nltk_df2_grupped['text']:
        id_all = []
        for sent_grupped_deep in sent_grupped:
            for sent in final_df_lists1['Sentence']:
                for sent_deep in sent:
                    if re.search(r'\w*(' + str(sent_deep) + ')\w*', str(sent_grupped_deep)):
                        id = nltk_df2_grupped['id'][i]
                        id_all.append(id)
        sentence_id.at[i, 'id'] = id_all
        sentence_id.at[i, "sentence"] = sent_grupped
        print('sentence_id', sentence_id) # Sentences grupped by article ID, might be usefull
        i += 1
    ########################


    ######################## Creating sentence_polarity_distribution_plot table with  sentence sentiment analysis scores
    # grupped by coin names along with article ID's

    absa_scores_df = pd.DataFrame(columns = ['id', 'coins', 'sentiment_polarity'])

    i=0
    for k, v in absa_scores.items():
        ids = v[-1]
        ids = list(dict.fromkeys(ids))
        absa_scores_df.at[i, 'id'] = " ".join(str(x) for x in ids)
        absa_scores_df.at[i, 'coins'] = k
        absa_scores_df.at[i, 'sentiment_polarity'] = v[:-1]
        i+=1

    absa_scores_df["ommit"]=range(len(absa_scores_df)) #It's only for Bigquery to know the column names

    print("absa_scores_df", absa_scores_df) # sentence sentiment analysis scores grupped by coin names along with article ID's
    absa_scores_df.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv',
        index=False, header=True)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv'
    load(filename, 'sentence_polarity_distribution_plot')
    print("Przetłumaczony tekst przeanalizowany, zapisany jako sentence_polarity_hisogram_plot.csv i wrzucony do tabeli sentence_polarity_distribution_plot")
    ########################


    ######################## Here the sentence_polarity_histogram_plot is created and it contains calculations needed for creating box plot in Data Studio -> REGARDLES THE ARTICLE
    # Can be used to summarise conclusions from all articles

    feature_polarity_quantiles_df=pd.DataFrame(columns=["coin", "lower_quartile", "median", "upper_quartile", "min", "max"])
    feature_polarity_quantiles_df["coin"] = absa_scores.keys()
    i=0
    for v in absa_scores.values():
        scores = []

        score = np.quantile(v[:-1], .25)
        scores.append(score)

        score = np.quantile(v[:-1], .50)
        scores.append(score)

        score = np.quantile(v[:-1], .75)
        scores.append(score)

        score = np.min(v[:-1])
        scores.append(score)

        score = np.max(v[:-1])
        scores.append(score)

        feature_polarity_quantiles_df.iloc[[i],[1]]=scores[0]
        feature_polarity_quantiles_df.iloc[[i],[2]]=scores[1]
        feature_polarity_quantiles_df.iloc[[i],[3]]=scores[2]
        feature_polarity_quantiles_df.iloc[[i],[4]]=scores[3]
        feature_polarity_quantiles_df.iloc[[i],[5]]=scores[4]
        i+=1


    print("feature_polarity_quantiles_df", feature_polarity_quantiles_df)
    feature_polarity_calculations_df=pd.DataFrame(columns=["coin", "min", "delta_lower_quartile", "median", "delta_upper_quartile", "delta_max"] )
    feature_polarity_calculations_df["coin"]=feature_polarity_quantiles_df["coin"]
    feature_polarity_calculations_df["min"]=feature_polarity_quantiles_df["min"]
    feature_polarity_calculations_df["median"]=feature_polarity_quantiles_df["median"]
    feature_polarity_calculations_df["delta_lower_quartile"]=feature_polarity_quantiles_df["lower_quartile"]-abs(feature_polarity_quantiles_df["min"])
    feature_polarity_calculations_df["delta_upper_quartile"]=feature_polarity_quantiles_df["upper_quartile"]-abs(feature_polarity_quantiles_df["lower_quartile"])
    feature_polarity_calculations_df["delta_max"]=feature_polarity_quantiles_df["max"]-abs(feature_polarity_quantiles_df["upper_quartile"])
    feature_polarity_calculations_df["max"]=feature_polarity_quantiles_df["max"]


    print("feature_polarity_calculations_df", feature_polarity_calculations_df)
    feature_polarity_calculations_df.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df.csv',
        index=False, header=True)
    filename='C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df.csv'
    load(filename, 'sentence_polarity_hisogram_plot_all_articles')
    print("Przetłumaczony tekst przeanalizowany, zapisany jako feature_polarity_calculations_df.csv i wrzucony do tabeli sentence_polarity_hisogram_plot_all_articles")
    #########################


    ######################## Here the sentence_polarity_histogram_plot_by_article is created and it contains calculations needed for creating box plot in Data Studio -> calculated per article ID
    def histogram_data_plot(id, j):
        feature_polarity_quantiles_df.at[j, 'id'] = id

        scores = []

        score = np.quantile(article_sent_scores, .25)
        scores.append(score)

        score = np.quantile(article_sent_scores, .50)
        scores.append(score)

        score = np.quantile(article_sent_scores, .75)
        scores.append(score)

        score = np.min(article_sent_scores)
        scores.append(score)

        score = np.max(article_sent_scores)
        scores.append(score)

        feature_polarity_quantiles_df.iloc[[j], [1]] = scores[0]
        feature_polarity_quantiles_df.iloc[[j], [2]] = scores[1]
        feature_polarity_quantiles_df.iloc[[j], [3]] = scores[2]
        feature_polarity_quantiles_df.iloc[[j], [4]] = scores[3]
        feature_polarity_quantiles_df.iloc[[j], [5]] = scores[4]

        feature_polarity_calculations_df = pd.DataFrame(
            columns=["id", "min", "delta_lower_quartile", "median", "delta_upper_quartile", "delta_max"])
        feature_polarity_calculations_df["id"] = feature_polarity_quantiles_df["id"]

        feature_polarity_calculations_df["min"] = feature_polarity_quantiles_df["min"]
        feature_polarity_calculations_df["median"] = feature_polarity_quantiles_df["median"]
        feature_polarity_calculations_df["lower_quartile"] = feature_polarity_quantiles_df["lower_quartile"]

        feature_polarity_calculations_df["delta_lower_quartile"] = feature_polarity_quantiles_df["lower_quartile"] - \
                                                                   abs(feature_polarity_quantiles_df["min"])
        feature_polarity_calculations_df["upper_quartile"] = feature_polarity_quantiles_df["upper_quartile"]

        feature_polarity_calculations_df["delta_upper_quartile"] = feature_polarity_quantiles_df["upper_quartile"] - \
                                                                   abs(feature_polarity_quantiles_df["lower_quartile"])
        feature_polarity_calculations_df["delta_max"] = feature_polarity_quantiles_df["max"] - \
                                                        abs(feature_polarity_quantiles_df["upper_quartile"])
        feature_polarity_calculations_df["max"] = feature_polarity_quantiles_df["max"]

        print("feature_polarity_calculations_df", feature_polarity_calculations_df)
        feature_polarity_calculations_df.to_csv(
            rf'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df_{j}.csv',
            index=False, header=True)


    feature_polarity_quantiles_df = pd.DataFrame(
        columns=["id", "lower_quartile", "median", "upper_quartile", "min", "max"])
    article_sent_scores = []
    i=0
    j=0
    for sent in nltk_df2['text']:
        row_id = nltk_df2[nltk_df2['text'] == sent].index[0]
        id = nltk_df2['id'][row_id]
        blob = TextBlob(sent)
        score = blob.sentiment.polarity
        article_sent_scores.append(score)
        i += 1
        if j == len(nltk_df2):
            histogram_data_plot(id, j)
        else:
            try:
                if id !=nltk_df2['id'][i]:
                    histogram_data_plot(id, j)
                    j += 1
                    article_sent_scores = []
            except:
                histogram_data_plot(id, j)

    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df_4.csv'
    load(filename, 'sentence_polarity_hisogram_plot_by_article')
    print("Przetłumaczony tekst przeanalizowany, zapisany jako feature_polarity_calculations_df.csv i wrzucony do tabeli sentence_polarity_hisogram_plot_by_article")
########################


