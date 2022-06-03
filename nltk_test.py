import nltk  # Python library for NLP
import matplotlib.pyplot as plt  # library for visualization
import json
import pandas as pd
import numpy as np
from Crypto_nlp.utils import process_text, get_dict, get_batches, compute_pca
import gensim
from gensim.models import Word2Vec, KeyedVectors
# For visualization of word2vec model
from sklearn.manifold import TSNE
import os
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

################################
#UNSUPERVISED MODEL

pd.options.display.max_colwidth = 500000000000
file = (
    "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng_ja_edytowalem_recznie.json")
records = map(json.loads, open(file, encoding="utf8"))
# df = pd.read_csv('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng.csv')
df = pd.DataFrame.from_records(records)
# text = df.loc[1:, ["text"]]
text=df['text']
print(text)

text = text.to_string()
# text = 'Twoja stara jest bardzo stara, bardzo, bardzo'
# print(text)
text = text.lower()

# text preprocessing
text_clean = process_text(text)

print(text_clean)

# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist = nltk.FreqDist(word for word in text_clean)
# print("Size of vocabulary: ",len(fdist))
fdist.pprint()
print("Most frequent tokens: ",fdist.most_common(20) ) # print the 20 most frequent words and their freq.


word2Ind, Ind2word = get_dict(text_clean)
V = len(word2Ind)
# print("Size of vocabulary: ", V)
# print("Index of the word 'index' : ", word2Ind['index'])
# print("Word which has index 198: ", Ind2word[198])


def initialize_model(N, V, random_seed=1):
    '''
    Inputs:
    N: dimension of hidden vector
    V: dimension of vocabulary
    random_seed: random seed for consistent results in the unit tests
    Outputs:
    W1, W2, b1, b2: initialized weights and biases
    '''
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    np.random.seed(random_seed)
    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)
    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)
    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)
    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)
    ### END CODE HERE ###
    return W1, W2, b1, b2


# test function:
tmp_N = 4
tmp_V = 10
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
assert tmp_W1.shape == ((tmp_N, tmp_V))
assert tmp_W2.shape == ((tmp_V, tmp_N))


# print(f"tmp_W1.shape: {tmp_W1.shape}")
# print(f"tmp_W2.shape: {tmp_W2.shape}")
# print(f"tmp_b1.shape: {tmp_b1.shape}")
# print(f"tmp_b2.shape: {tmp_b2.shape}")
#
#
def softmax(z):
    '''
    Inputs:
    z: output scores from the hidden layer
    Outputs:
    yhat: prediction (estimate of y)
    '''
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate yhat (softmax)
    # yhat can be interpreted as the probability of being a center word
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    ### END CODE HERE ###
    return yhat

# Test the function
tmp = np.array([[1, 2, 3],
                [1, 1, 1]
                ])
tmp_sm = softmax(tmp)
# print(tmp_sm)

# arr=softmax([9,8,11,10,8.5])

# Forward propagation to tak naprawde trenowanie modelu. X to
def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs:
    x: average one hot vector for the context
    W1, W2, b1, b2: matrices and biases to be learned
    Outputs:
    z: output score vector
    '''
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate h
    h = np.dot(W1, x) + b1
    # Apply the RELU on h (store result in h)
    h = np.maximum(0, h)
    # Calculate z
    z = np.dot(W2, h) + b2
    ### END CODE HERE ###
    return z, h


# Test the function
# Create some inputs
# tmp_N = 2
# tmp_V = 3
# tmp_x = np.array([[0, 1, 0]]).T
# # print(tmp_x)
# tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(N=tmp_N, V=tmp_V, random_seed=1)
# # print(f"x has shape {tmp_x.shape}")
# # print(f"N is {tmp_N} and vocabulary size V is {tmp_V}")
# # call function
# tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
#
#
# print("call forward_prop")
#
# print(f"z has shape {tmp_z.shape}")
# print("z has values:")
# print(tmp_z)
# print()
# print(f"h has shape {tmp_h.shape}")
# print("h has values:")
# print(tmp_h)
#
def compute_cost(y, yhat, batch_size):
    # cost function
    logprobs = np.multiply(np.log(yhat), y)
    cost = - 1 / batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


# Test the function
# tmp_C = 2
# tmp_N = 50
# tmp_batch_size = 4
# tmp_word2Ind, tmp_Ind2word = get_dict(text_clean)
# tmp_V = len(word2Ind)
# tmp_x, tmp_y = next(get_batches(text_clean, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))
# print(f"tmp_x.shape {tmp_x.shape}")
# print(f"tmp_y.shape {tmp_y.shape}")
# tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
# print(f"tmp_W1.shape {tmp_W1.shape}")
# print(f"tmp_W2.shape {tmp_W2.shape}")
# print(f"tmp_b1.shape {tmp_b1.shape}")
# print(f"tmp_b2.shape {tmp_b2.shape}")
# tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
# print(f"tmp_z.shape: {tmp_z.shape}")
# print(f"tmp_h.shape: {tmp_h.shape}")
# tmp_yhat = softmax(tmp_z)
# print(f"tmp_yhat.shape: {tmp_yhat.shape}")
# tmp_cost = compute_cost(tmp_y, tmp_yhat, tmp_batch_size)
# print("call compute_cost")
# print(f"tmp_cost {tmp_cost:.4f}")
#
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs:
    x: average one hot vector for the context
    yhat: prediction (estimate of y)
    y: target vector
    h: hidden vector (see eq. 1)
    W1, W2, b1, b2: matrices and biases
    batch_size: batch size
    Outputs:
    grad_W1, grad_W2, grad_b1, grad_b2: gradients of matrices and biases
    '''
    ### START CODE HERE (Replace instanes of 'None' with your code) ###
    # Compute l1 as W2^T (Yhat - Y)
    # and re-use it whenever you see W2^T (Yhat - Y) used to compute a gradient
    l1 = np.dot(W2.T, (yhat - y))
    # Apply relu to l1
    l1 = np.maximum(0, l1)
    # Compute the gradient of W1
    grad_W1 = np.dot(l1, x.T) / batch_size
    # Compute the gradient of W2
    grad_W2 = np.dot(yhat - y, h.T) / batch_size
    # Compute the gradient of b1
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    # Compute the gradient of b2
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size
    ### END CODE HERE ####
    return grad_W1, grad_W2, grad_b1, grad_b2
#
#
# Test the function
tmp_C = 2
tmp_N = 50
tmp_batch_size = 4
tmp_word2Ind, tmp_Ind2word = get_dict(text_clean)
tmp_V = len(word2Ind)
# get a batch of data
tmp_x, tmp_y = next(get_batches(text_clean, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))
# print("get a batch of data")
# print(f"tmp_x.shape {tmp_x.shape}")
# print(f"tmp_y.shape {tmp_y.shape}")
# print()
# print("Initialize weights and biases")
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
# print(f"tmp_W1.shape {tmp_W1.shape}")
# print(f"tmp_W2.shape {tmp_W2.shape}")
# print(f"tmp_b1.shape {tmp_b1.shape}")
# print(f"tmp_b2.shape {tmp_b2.shape}")
# print()
# print("Forwad prop to get z and h")
tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
# print(f"tmp_z.shape: {tmp_z.shape}")
# print(f"tmp_h.shape: {tmp_h.shape}")
# print()
# print("Get yhat by calling softmax")
tmp_yhat = softmax(tmp_z)
# print(f"tmp_yhat.shape: {tmp_yhat.shape}")

tmp_m = (2 * tmp_C)
tmp_grad_W1, tmp_grad_W2, tmp_grad_b1, tmp_grad_b2 = back_prop(tmp_x, tmp_yhat, tmp_y, tmp_h, tmp_W1, tmp_W2, tmp_b1,
                                                               tmp_b2, tmp_batch_size)
#

print()
print("call back_prop")
print(f"tmp_grad_W1.shape {tmp_grad_W1.shape}")
print(f"tmp_grad_W2.shape {tmp_grad_W2.shape}")
print(f"tmp_grad_b1.shape {tmp_grad_b1.shape}")
print(f"tmp_grad_b2.shape {tmp_grad_b2.shape}")

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03,
                     random_seed=282, initialize_model=initialize_model,
                     get_batches=get_batches, forward_prop=forward_prop,
                     softmax=softmax, compute_cost=compute_cost,
                     back_prop=back_prop):
    '''
    This is the gradient_descent function
    Inputs:
    data: text
    word2Ind: words to Indices
    N: dimension of hidden vector
    V: dimension of vocabulary
    num_iters: number of iterations
    random_seed: random seed to initialize the model's matrices and vectors
    initialize_model: your implementation of the function to initialize the␣
    ,!model
    get_batches: function to get the data in batches
    forward_prop: your implementation of the function to perform forward␣
    ,!propagation
    softmax: your implementation of the softmax function
    compute_cost: cost function (Cross entropy)
    back_prop: your implementation of the function to perform backward␣
    ,!propagation
    Outputs:
    W1, W2, b1, b2: updated matrices and biases after num_iters iterations
    '''
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=random_seed)  # W1=(N,V),and W2=(V,N)
    batch_size = 128
    # batch_size = 512
    iters = 0
    C = 2
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        ### START CODE HERE (Replace instances of 'None' with your own code),#### get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # Get yhat
        yhat = softmax(z)
        # Get cost
        cost = compute_cost(y, yhat, batch_size)
        if ((iters + 1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        # get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        # update weights and biases
        W1 -= alpha * grad_W1
        W2 -= alpha * grad_W2
        b1 -= alpha * grad_b1
        b2 -= alpha * grad_b2
        ### END CODE HERE ###
        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66
    return W1, W2, b1, b2


# test your function
# # UNIT TEST COMMENT: Each time this cell is run the cost for each iteration,changes slightly (the change is less dramatic after some iterations)
# # to have this into account let's accept an answer as correct if the cost of, iter 15 = 41.6 (without caring about decimal points beyond the first decimal)
# 41.66, 41.69778, 41.63, etc should all be valid answers.
C = 2
N = 300
word2Ind, Ind2word = get_dict(text_clean)
V = len(word2Ind)
num_iters = 140
W1, W2, b1, b2 = gradient_descent(text_clean, word2Ind, N, V, num_iters)
print("Call gradient_descent")

# visualizing the word vectors here

# words = ['index', 'token','market','project', 'bitcoin','ethereum','crypto',
# 'weight','asset']
words = ['litecoin', 'polkadot', 'bitcoin cash', 'stellar', 'dogecoin', 'binance coin', 'tether', 'monero', 'solana',
         'avalanche', 'usd coin', 'chainlink', 'Algorand', 'polygon', 'vechain', 'tron', 'zcash', 'eos', 'tezos', 'neo',
         'stacks', 'nem', 'decred', 'storj', '0x', 'digibyte', 'index', 'token', 'crypto', 'market', 'project',
         'bitcoin']
# extracting embeddings
embs = (W1.T + W2) / 2.0
# given a list of words and the embeddings, it returns a matrix with all the, embeddings

idx = [word2Ind[word] for word in words]
idx = []
for word in words:
    if word in word2Ind:
        idx.append(word2Ind[word])
    else:
        idx.append(0)
        print(f'there is no "{word}"')

X = embs[idx, :]
print(X.shape)
print(type(X))
print(X.shape, idx) # X.shape: Number of words of dimension N each

result= compute_pca(X, 2)
print(result)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

# Logika działania algorytmu
# 1. Preprocessing danych -> remove hashtags, tokenizacja, remove stopwords, remove punctuation, stemming word
# 2. Compute the frequency distribution of the words in the dataset (vocabulary) -> Po CO?
# 3. Jak obliczamy X -> get vectors ma window function, tworzy x rozkmin dalej c2_w4_lecture_nb_3
# 4. Po co liczym word2ind i na odwrot -> Chyba tylko po to zeby dalo sie wyliczyc gradient descent



####################################

#PRETRAINED WORD2VEC

# pretrained_google_news_model = KeyedVectors.load_word2vec_format(
#     'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield line.split()


from gensim import utils
from gensim.test.utils import datapath


# class MyCorpus(object):
#     """An interator that yields sentences (lists of str)."""
#
#     def __iter__(self):
#         corpus_path = datapath(
#             'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng.txt')
#         for line in open(corpus_path):
#             # assume there's one document per line, tokens separated by whitespace
#             yield utils.simple_preprocess(line)


# sentences = MyCorpus()
sentences = ['litecoin', 'polkadot', 'bitcoin cash', 'stellar', 'dogecoin', 'binance coin', 'tether', 'monero', 'solana',
      'avalanche', 'usd coin', 'chainlink', 'Algorand', 'polygon', 'vechain', 'tron', 'zcash', 'eos', 'tezos', 'neo',
      'stacks', 'nem', 'decred', 'storj', '0x', 'digibyte', 'index', 'token', 'crypto', 'market', 'project',
      'bitcoin']
# model_cbow = Word2Vec(sentences, min_count=1)
preprocessed_text = text.apply(gensim.utils.simple_preprocess)

# model_cbow = Word2Vec(preprocessed_text)
# model_cbow.build_vocab(corpus_iterable=preprocessed_text, update=True)
# model_cbow = Word2Vec(preprocessed_text, min_count=1, workers=4, vector_size=200,  window =3, epochs=20)
# print(model_cbow.corpus_count)
# model_cbow.train(
#     list(df), total_examples=model_cbow.corpus_count, epochs=model_cbow.epochs)
# model_cbow.wv.save_word2vec_format("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec2.bin", binary=True)

# model_cbow = Word2Vec.load('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec2.gz')

# print(model_cbow.wv.evaluate_word_analogies('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec'))

# print(model_cbow.wv.most_similar(['polkadot']))
# c=model_cbow.wv.most_similar(['bitcoin'])
# model = gensim.models.KeyedVectors.load('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec2.bin', mmap='r')
# model = Word2Vec.load('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec2.bin')
model = gensim.models.keyedvectors.load_word2vec_format('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/word2vec/custom_word2vec2.bin', binary=True)
# print(model['computer'])
# print(model['bitcoin'])
print(model.n_similarity(['bitcoin'], ['solana']))
# most_similar=model.similar_by_word(word_labels[1])
# print([i[0] for i in most_similar])



# def display_closestwords_tsnescatterplot(model, word):
#     # arr = np.empty((8, 200), dtype='f')
#     w = list(filter(lambda x: x in model.key_to_index, sentences)) #filtruje slowa z listy na podstawie vocab
#     word_labels = w
#     # get close words
#     close_words=[]
#     wrd_vector=[]
#     for i in range(len(w[0:2])):
#         # close_words.append(model.similar_by_word(w[i]))
#         close_words.append(model.most_similar(positive=w))
#         # close_words = np.append(close_words, np.array([i[0] for i in most_similar]), axis=0)
#     print('close_words:', close_words)
#     # add the vector for each of the closest words to the array
#     z = list(filter(lambda x: x in model.key_to_index, [i[0] for i in close_words[0]]))
#     print('z:',z)
#     print('model[w]:',np.array(model[w]))
#     arr = model[w]
#     print('arr', arr)
#     for i in range(len(z)):
#         wrd_vector.append(model[z[i]])
#         word_labels.append(z[i])
#     # print('wrd_vector:', wrd_vector)
#     # print('word_labels:', word_labels)
#     # arr = np.append(arr, [wrd_vector], axis=0)
#         # print('i:', i)
#     # find tsne coords for 2 dimensions
#     tsne = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(arr)
#
#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     # display scatter plot
#     plt.scatter(x_coords, y_coords)
#
#     for label, x, y in zip(word_labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
#     plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
#     plt.show()
# #
# display_closestwords_tsnescatterplot(model, sentences)



###############################
#TESTOWY KOD
# close_words = []
# w = list(filter(lambda x: x in model.key_to_index, sentences))
# for i in range(len(w)):
#     close_words.append(model.most_similar(positive=w))
#     # close_words.append(model.similar_by_word(w[i]))
# print('closeword:',close_words)
# print('w', w)
# z = list(filter(lambda x: x in model.key_to_index, [i[0] for i in close_words[0]]))
# print('z', z)
#
# # for i in range(len(z)):
# #     wrd_vector = model[z[i]]
# # print([wrd_score[0] for wrd_score in close_words[0]])
#
# # print('close_words:',close_words)
# # print('wrd_vector:',wrd_vector)
# # print(w)
# # print(f"word in vocab:{i}")
# # print(model.most_similar(positive=w))
# # word_labels =[]
# # # words= [i[0] for i in close_words]
# # # words = close_words
# arr = np.empty((len(close_words[0]),200), dtype='f')
# word_labels = [sentences]
# # print(model['bitcoin'].shape)
# # print(arr.shape)
# for i in range(len(w)):
#     np.append(arr, np.array([model[w[i]]]), axis=0)
#
# z = list(filter(lambda x: x in model.key_to_index, [i[0] for i in close_words[0]]))
# for i in range(len(z)):
#     wrd_vector = model[z[i]]
#     arr = np.append(arr, np.array([wrd_vector]), axis=0)
#     # print("wrd_score:", wrd_score)
#     # print("wrd_vector:", wrd_vector.shape)
#
#     # print("wrd_vector:", wrd_vector)
#     # print("arr:", arr)
# word_labels.append(close_words[0])
# # print("close_words",len(close_words[0]))
# # print("wrd_vector:", wrd_vector.shape)
# # print("arr:", arr.shape)
# # print("word_labels:", word_labels)
# # for i in range(len(close_words[0])):
# #     w = list(filter(lambda x: x in model.key_to_index, close_words[0]))
# # print(w)
##########################################