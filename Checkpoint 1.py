# Input: There is a financial institution near the river bank.

# Taking a sentence as input
sentence = input("Enter A Sentence: ")

# Tokenizing the sentence into words
from nltk.tokenize import word_tokenize
words = word_tokenize(sentence)

# Lemmatizing the tokenized words
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for i in words:
    lemmatizer.lemmatize(i)

# Removing stop-words from the list
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

# Extracting relevant words from the lexical database - WordNet
from nltk.corpus import wordnet
for word in words:
    if not wordnet.synsets(word):
        words.remove(word)

# Word Similarity Matrix (Co-occurrence graph)
word_similarity_matrix = [[0 for i in range(len(words))] for j in range(len(words))]
for i in range(0, len(words)):
    for j in range(0, len(words)):
        if i == j:
            continue

        # Dice coefficient
        bigram1 = set(words[i])
        bigram2 = set(words[j])
        word_similarity_matrix[i][j] = len(bigram1 & bigram2) * 2.0 / (len(bigram1) + len(bigram2))

# N-Gram graph
n_gram_graph = [(words[i], words[i+1]) for i in range(0, len(words)-1)]

# Computing Mean of the co-occurrence graph
avg = 0
for i in range(0, len(words)):
    for j in range(0, len(words)):
        avg += word_similarity_matrix[i][j]
avg = avg / (len(words) * len(words))

# Similarity n-gram graph
for i in range(0, len(words)):
    for j in range(0, len(words)):
        if i == j:
            continue
        if (words[i], words[j]) in n_gram_graph or (words[j], words[i]) in n_gram_graph:
            word_similarity_matrix[i][j] += avg

'''
# Extracting senses of each word
sense_matrix = []
for word in words:
    word_senses = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            word_senses.append(lemma.name())
    unique_word_senses = []
    for sense in word_senses:
        if not sense in unique_word_senses:
            unique_word_senses.append(sense)
    if not len(unique_word_senses) <= 5:    # Number of unique senses = 5 (assumption)
        unique_word_senses = unique_word_senses[0:5]
    sense_matrix.append(unique_word_senses)
'''

# Extracting synsets of each word
synsets = []
sense_count = []
for word in words:
    word_synsets = []
    for synset in wordnet.synsets(word):
        word_synsets.append(synset)
    unique_word_synsets = []
    for synset in word_synsets:
        if not synset in unique_word_synsets:
            unique_word_synsets.append(synset)
    if not len(unique_word_synsets) <= 5:   # Number of unique senses = 5 (assumption)
        unique_word_synsets = unique_word_synsets[0:5]
    for synset in unique_word_synsets:
        synsets.append(synset)
    sense_count.append(len(unique_word_synsets))

# Sense similarity matrix
sense_similarity_matrix = [[0 for i in range(len(synsets))] for j in range(len(synsets))]
for i in range(0, len(synsets)):
    for j in range(0, len(synsets)):
        similarity = synsets[i].wup_similarity(synsets[j])
        if similarity == None:
            similarity = 0
        sense_similarity_matrix[i][j] = similarity

# Strategy Space
strategy_space = [[0 for i in range(len(synsets))] for j in range(len(words))]

# Assigning a normal distribution to the strategy space
count = 0
for i in range(0, len(words)):
    for j in range(0, sense_count[i]):
        strategy_space[i][j+count] = 1 / sense_count[i]
    count += sense_count[i]



# Input: There is a financial institution near the river bank.