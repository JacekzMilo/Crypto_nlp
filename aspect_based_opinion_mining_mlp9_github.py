
def nlp_article_semantic(file):
    from Crypto_nlp.Load_to_GCP import load
    from time import sleep
    from textblob import TextBlob, Word
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    import nltk
    import json
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    #
    # txt = (
    #     f"{file}")
    # records = map(json.loads, open(txt, encoding="utf8"))
    # df = pd.DataFrame.from_records(records)
    # result=df["text"]
    df = pd.read_csv(f"{file}", encoding="utf8")
    # print("df", df)
    # print("df['article_text_translated']", df["article_text_translated"])
    result = df["article_text_translated"]
########################
# text cleaning
    from string import punctuation
    import re

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
    ########################
    result = [clean_sentence(x) for x in result]
    print('words', result)


    # Check sentiment polarity of each sentence.
    sentiment_scores = list()
    i = 0
    for sentence in result:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        if(i <= 10):
            print(sentence + ": POLARITY=" + str(line.sentiment.polarity))
            i += 1

    # print(sns.distplot(sentiment_scores))

    # Convert array of comments into a single string
    comments = TextBlob(' '.join(result))
    # Check out noun phrases, will be useful for frequent feature extraction
    print('comments.noun_phrases', comments.noun_phrases)

    featureList=['litecoin', 'polkadot', 'bitcoin', 'stellar', 'dogecoin', 'binance', 'tether', 'monero', 'solana',
             'avalanche', 'chainlink', 'algorand', 'polygon', 'vechain', 'tron', 'zcash', 'eos', 'tezos', 'neo',
             'stacks', 'nem', 'decred', 'storj', '0x', 'digibyte']

    # # compactness pruning:
    # cleaned = list()
    # for phrase in featureList:
    #     count = 0
    #     for word in phrase.split():
    #         # Count the number of small words and words without an English definition
    #         if len(word) <= 2 or (not Word(word).definitions):
    #             count += 1
    #     # Only if the 'nonsensical' or short words DO NOT make up more than 40% (arbitrary) of the phrase add
    #     # it to the cleaned list, effectively pruning the ones not added.
    #     if count < len(phrase.split()) * 0.4:
    #         cleaned.append(phrase)
    #
    # print("After compactness pruning:\nFeature Size:")
    # print(len(cleaned))
    #
    # for phrase in cleaned:
    #     match = list()
    #     temp = list()
    #     word_match = list()
    #     for word in phrase.split():
    #         # Find common words among all phrases
    #         word_match = [p for p in cleaned if re.search(word, p) and p not in word_match]
    #         # If the size of matched phrases set is smaller than 30% of the cleaned phrases,
    #         # then consider the phrase as non-redundant.
    #         if len(word_match) <= len(cleaned) * 0.3:
    #             temp.append(word)
    #             match += word_match
    #
    #     phrase = ' '.join(temp)
    #     #     print("Match for " + phrase + ": " + str(match))
    #
    #     if len(match) >= len(cleaned) * 0.1:
    #         # Redundant feature set, since it contains more than 10% of the number of phrases.
    #         # Prune all matched features.
    #         for feature in match:
    #             if feature in cleaned:
    #                 cleaned.remove(feature)
    #
    #         # Add largest length phrase as feature
    #         cleaned.append(max(match, key=len))
    #
    # print("After redundancy pruning:\nFeature Size:" + str(len(cleaned)))
    # print("Cleaned features:")
    # print(cleaned)

    from nltk.corpus import stopwords

    feature_count = dict()
    for phrase in featureList:
        count = 0
        for word in phrase.split():
            if word not in stopwords.words('english'):
                count += comments.words.count(word)

        print(phrase + ": " + str(count))
        feature_count[phrase] = count

    # Select frequent feature threshold as (max_count)/100
    # This is an arbitrary decision as of now.
    counts = list(feature_count.values())
    features = list(feature_count.keys())
    threshold = len(featureList)/100
    #threshold=66

    print("Threshold:" + str(threshold))

    frequent_features = list()
    # tu okresla ktore coiny sa najczestsze
    for feature, count in feature_count.items():
        if count >= threshold:
            frequent_features.append(feature)
    print(' Features:')
    frequent_features=frequent_features[0:5]
    print(frequent_features)

    # nltk.download('vader_lexicon')

    def nltk_sentiment(sentence):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk_sentiment = SentimentIntensityAnalyzer()
        score = nltk_sentiment.polarity_scores(sentence)
        return score

    ############# tu robi analize dla poszczegolnych slow kluczowych, nie uzwam.
    # b=dataset.values.T.tolist()
    # print(b)
    nltk_results = [nltk_sentiment(row) for row in frequent_features]
    # print(nltk_results)
    results_df = pd.DataFrame(nltk_results)
    # print(results_df)
    text_df = pd.DataFrame(frequent_features)
    # print(text_df)
    nltk_df = text_df.join(results_df)
    # nltk_df1=nltk_df[[0,'neu']]
    # print('nltk_df.head(5)', nltk_df.head(5))
    #############

    newdf=nltk_df[0]
    newdf=pd.DataFrame({'features':nltk_df[0],'pos':nltk_df['pos'],'neg':nltk_df['neg']})
    newdf.pos=newdf.pos+0.2
    newdf.neg=newdf.neg-0.2
    # print('newdf', newdf)


    absa_list = dict()
    sentences=[]
    # For each frequent feature
    for f in frequent_features:
        # For each comment
        absa_list[f] = list()
        for comment in result:
            blob = TextBlob(comment)
            # For each sentence of the comment
            for sentence in blob.sentences:
                # Search for frequent feature 'f'
                q = '|'.join(f.split())
                if re.search(r'\w*(' + str(q) + ')\w*', str(sentence)):
                    absa_list[f].append(sentence)
                    sentences.append(sentence)


    # print("sentences", sentences)
    sentences=" ".join(str(x) for x in sentences)
    from nltk.tokenize import sent_tokenize
    sentences=sent_tokenize(sentences)

    # print('sentences', sentences)
    # print("absa_list[f]", absa_list.values())

    ###################### tu jest analiza sentymentu zdan w odniesieniu do slow kluczowych i wrzuca DF do BQ w tabele results_for_plot
    nltk_results = [nltk_sentiment(row) for row in sentences]
    # print("nltk_results", nltk_results)
    # print("nltk_results[1]", nltk_results[1])
    results_df = pd.DataFrame(nltk_results)
    text_df = pd.DataFrame(sentences)
    nltk_df2=pd.DataFrame()
    nltk_df2['text'] = text_df
    nltk_df2=nltk_df2.join(results_df)
    print('nltk_df.head(5)2', nltk_df2.head(5))

    nltk_df2.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv',
        index=False, header=True)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv'
    load(filename, 'results_for_plot')
    ######################

    scores = list()
    absa_scores = dict()
    for k, v in absa_list.items():
        # print("k", k)
        absa_scores[k] = list()
        # print("v", v)
        for sent in v:
            score = sent.sentiment.polarity #to jest dla zdania
            scores.append(score)
            absa_scores[k].append(score)
            print("sent", sent)

    print("absa_list", absa_list)
    litecoin_list=[]
    polkadot_list=[]
    bitcoin_list=[]
    stellar_list=[]
    dogecoin_list=[]
    binance_list=[]
    tether_list=[]
    monero_list=[]
    solana_list=[]
    avalanche_list=[]
    chainlink_list=[]
    algorand_list=[]
    polygon_list=[]
    vechain_list=[]
    tron_list=[]
    zcash_list=[]
    eos_list=[]
    tezos_list=[]
    neo_list=[]
    stacks_list=[]
    nem_list=[]
    decred_list=[]
    storj_list=[]
    x_list=[]
    digibyte_list=[]
    sent_all=[]

    # print('sentences', sentences)
    frequent_features_str=" ".join(str(x) for x in frequent_features)

    # final_dic={"Coin": [], "Sentence": [], "Sentence sentiment analysis": []}
    # final_df=pd.DataFrame.from_dict(final_dic)

    i=0
    for sent in sentences:
        # print("sent", sent)
        for coin in frequent_features:
            # print("coin", coin)
            if coin in sent:
                if coin == "bitcoin":
                    bitcoin_list.append(nltk_results[i])
                    sent2 = [coin]
                    sent_all.append([sent])
                    sent2.append(sent_all)
                    sent2.append(bitcoin_list)

                if coin == "polkadot":
                    polkadot_list.append(nltk_results[i])
                    sent3 = [coin]
                    sent3.append([sent])
                    sent3.append(polkadot_list)

                if coin == "chainlink":
                    chainlink_list.append(nltk_results[i])
                    sent4 = [coin]
                    sent4.append([sent])
                    sent4.append(chainlink_list)

        i += 1
    final_list=[]

    final_list.append(sent2)
    final_list.append(sent3)
    final_list.append(sent4)

    df_scores_polarity=pd.DataFrame(scores, columns=["sentence_polarity"])
    print("sentence_polarity", df_scores_polarity)

    final_df=pd.DataFrame(final_list, columns=['Coin', 'Sentence', 'Sentence sentiment analysis' ])
    final_df["ommit"]=range(len(frequent_features)) #to jest tylko po to zeby BQ rozpoznawal nazwy kolumn
    # print("range", range(len(frequent_features)))
    print("sentiment_scores", sentiment_scores)
    print('frequent_features', frequent_features)
    print("Aspect Specific sentences:")
    print(absa_list)
    print("final_df:")
    print(final_df)
    # print("nltk_df")
    # print(nltk_df)
    print("scores", scores)
    print("absa_scores", absa_scores)
    # print("newdf", newdf)
    # df_absa_scores=pd.DataFrame.from_dict(absa_scores)
    # print("df_absa_scores", df_absa_scores)
    # print(absa_list.values())

    ################################
    #plot
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # noun=['positive attitude','good job knowledge','team player','customer acquisition','good visibility','back office process','successful launch','soft skills','suitable candidates','core issues']
    # pos_l=[0.71,0.42,0.3,0.2,0.2,0.6,0.2,0.5,0.43,0]
    # neg_l=[-0.1,0,-0.300,0,-0.1,0,-0.1,-0.2,0,-0.3]

    # pos = nltk_df[0:5]['pos']
    # neg = nltk_df[0:5]['neg']

    # data to plot
    n_groups = len(frequent_features)
    positive = nltk_df2['pos'].head(len(frequent_features))
    negative = nltk_df2['neg'].head(len(frequent_features))

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 1

    rects1 = plt.bar(index, positive, bar_width,
                     alpha=opacity,
                     color='b',
                     label='positive sentiments')

    rects2 = plt.bar(index + bar_width, negative, bar_width,
                     alpha=opacity,
                     color='r',
                     label='negative sentiments')

    plt.xlabel('Features')
    plt.ylabel('sentiment value')
    plt.title('Top features and its sentiment')
    plt.xticks(index + bar_width, nltk_df2['text'].head(5))
    plt.legend()
    fig.set_size_inches(15, 10)
    plt.show()

    ##########################


    #########################
    #Plot 2

    # Now that we have all the scores, let's plot them!
    # For comparison, we replot the previous global sentiment polarity plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
    plot1 = sns.distplot(scores, ax=ax1)

    ax1.set_title('Aspect wise scores')
    ax1.set_xlabel('Sentiment Polarity')
    ax1.set_ylabel('# of comments')

    ax2.set_title('Article wise scores')
    ax2.set_xlabel('Sentiment Polarity')
    ax2.set_ylabel('# of comments')

    plot2 = sns.distplot(sentiment_scores, ax=ax2)
    plt.show()
    ##########################

    # Create data values for stripplot and boxplot
    vals = dict()
    vals["aspects"] = list()
    vals["scores"] = list()
    for k, v in absa_scores.items():
        # print('k',k)
        # print('v', v)
        for score in v:
            vals["aspects"].append(k)
            vals["scores"].append(score)
    print("vals", vals)
    print("vals[scores]", vals["scores"])

    #########################
    #Plot 3

    fig, ax1 = plt.subplots(figsize=(15, 10))
    color = sns.color_palette("Blues")
    plt.xticks(rotation=90)
    sns.set_context("paper", font_scale=3)
    sns.boxplot(x="aspects", y="scores", data=vals, palette=color, ax=ax1)

    #ten chyba najmniej istotny, inne pokazanie wykresu 1
    color = sns.color_palette("Blues", 1)
    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.xticks(rotation=90)
    sns.set_context("paper", font_scale=5)
    sns.stripplot(y="aspects", x="scores",data=vals, palette=color)
    plt.show()



#########################
    #to wszystko po to zeby stworzyc wykres boxplot w DS
    columns=["coins", "sentiment_polarity"]
    absa_scores_dict={"coins":absa_scores.keys(), "sentiment_polarity": absa_scores.values()}
    absa_scores_df=pd.DataFrame.from_dict(absa_scores_dict)
    # absa_scores_df["sentiment_polarity"].apply(', '.join)
    absa_scores_df["ommit"] = range(len(absa_scores_df))
    absa_scores_df.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv',
        index=False, header=True)
    # sleep(1)
    filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv'
    load(filename, 'sentence_polarity_distribution_plot')


    #Obliczanie kwartali zeby dalo sie narysowac boxplot w DS

    feature_polarity_quantiles=dict()
    feature_polarity_quantiles_df=pd.DataFrame(columns=["coin", "lower_quartile", "median", "upper_quartile", "min", "max"])
    feature_polarity_quantiles_df["coin"] = absa_scores.keys()
    i=0
    for v in absa_scores.values():
        scores = []

        score = np.quantile(v, .25)
        scores.append(score)

        score = np.quantile(v, .50)
        scores.append(score)

        score = np.quantile(v, .75)
        scores.append(score)

        score = np.min(v)
        scores.append(score)

        score = np.max(v)
        scores.append(score)
        # print("scores", scores)
        feature_polarity_quantiles_df.iloc[[i],[1]]=scores[0]
        feature_polarity_quantiles_df.iloc[[i],[2]]=scores[1]
        feature_polarity_quantiles_df.iloc[[i],[3]]=scores[2]
        feature_polarity_quantiles_df.iloc[[i],[4]]=scores[3]
        feature_polarity_quantiles_df.iloc[[i],[5]]=scores[4]

        i+=1


    # print("feature_polarity_quantiles_df", feature_polarity_quantiles_df)
    feature_polarity_calculations_df=pd.DataFrame(columns=["coin", "min", "delta_lower_quartile", "median", "delta_upper_quartile", "delta_max"] )
    feature_polarity_calculations_df["coin"]=feature_polarity_quantiles_df["coin"]
    feature_polarity_calculations_df["min"]=feature_polarity_quantiles_df["min"]
    feature_polarity_calculations_df["median"]=feature_polarity_quantiles_df["median"]
    feature_polarity_calculations_df["delta_lower_quartile"]=feature_polarity_quantiles_df["lower_quartile"]-feature_polarity_quantiles_df["min"]
    feature_polarity_calculations_df["delta_upper_quartile"]=feature_polarity_quantiles_df["upper_quartile"]-feature_polarity_quantiles_df["lower_quartile"]
    feature_polarity_calculations_df["delta_max"]=feature_polarity_quantiles_df["max"]-feature_polarity_quantiles_df["upper_quartile"]

    print("feature_polarity_calculations_df", feature_polarity_calculations_df)
    feature_polarity_calculations_df.to_csv(
        r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df.csv',
        index=False, header=True)
    filename='C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/feature_polarity_calculations_df.csv'
    load(filename, 'sentence_polarity_hisogram_plot')
#########################




