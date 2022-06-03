# uncomment if gensim is installed
# !pip install gensim
import gensim
# Need the interactive Tools for Matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

##################### oryginalny kod:
def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    print("close_words:", close_words)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    print('arr:', arr.shape)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        # print("wrd_vector:", wrd_vector.shape)
        # print("word_labels:", word_labels)
        # print("arr2:", arr)
        # print("word shape", model[word].shape)
        print("wrd_score", wrd_score)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model, 'tasty')
#######################

# arr = np.empty((0, 300), dtype='f')
# arr = np.append(arr, np.array([model['tasty']]), axis=0)
# print(arr.shape)
# print(model['tasty'].shape)


# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []
#
#     for word in model.wv.key_to_index:
#         tokens.append(model.wv[word])
#         labels.append(word)
#
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)
#
#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
#
#     plt.figure(figsize=(16, 16))
#     for i in range(len(x)):
#         plt.scatter(x[i], y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()

# from sklearn.decomposition import PCA
# import seaborn as sns
#
# def tsne_plot(for_word, w2v_model):
#     # trained word2vec model dimension
#     dim_size = w2v_model.wv.vectors.shape[1]
#
#     arrays = np.empty((0, dim_size), dtype='f')
#     word_labels = [for_word]
#     color_list = ['red']
#
#     # adds the vector of the query word
#     arrays = np.append(arrays, w2v_model.wv.__getitem__([for_word]), axis=0)
#
#     # gets list of most similar words
#     sim_words = w2v_model.wv.most_similar(for_word, topn=10)
#
#     # adds the vector for each of the closest words to the array
#     for wrd_score in sim_words:
#         wrd_vector = w2v_model.wv.__getitem__([wrd_score[0]])
#         word_labels.append(wrd_score[0])
#         color_list.append('green')
#         arrays = np.append(arrays, wrd_vector, axis=0)
#
#     # ---------------------- Apply PCA and tsne to reduce dimention --------------
#
#     # fit 2d PCA model to the similar word vectors
#     model_pca = PCA(n_components=10).fit_transform(arrays)
#
#     # Finds 2d coordinates t-SNE
#     np.set_printoptions(suppress=True)
#     Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(model_pca)
#
#     # Sets everything up to plot
#     df_plot = pd.DataFrame({'x': [x for x in Y[:, 0]],
#                             'y': [y for y in Y[:, 1]],
#                             'words_name': word_labels,
#                             'words_color': color_list})
#     # plot dots with color and position
#     plot_dot = sns.regplot(data=df_plot,
#                            x="x",
#                            y="y",
#                            fit_reg=False,
#                            marker="o",
#                            scatter_kws={'s': 40,
#                                         'facecolors': df_plot['words_color']
#                                         }
#                            )
#
#     # Adds annotations with color one by one with a loop
#     for line in range(0, df_plot.shape[0]):
#         plot_dot.text(df_plot["x"][line],
#                       df_plot['y'][line],
#                       '  ' + df_plot["words_name"][line].title(),
#                       horizontalalignment='left',
#                       verticalalignment='bottom', size='medium',
#                       color=df_plot['words_color'][line],
#                       weight='normal'
#                       ).set_size(15)
#
#     plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
#     plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)
#
#     plt.title('t-SNE visualization for word "{}'.format(for_word.title()) + '"')
#     plt.show()

# tsne_plot('bitcoin', model_cbow)
# print(c)