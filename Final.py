# Importing the relevant packages
import numpy as np

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

# Computing mean of the co-occurrence graph
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
word_count = 0
synsets = []
sense_count = np.zeros(len(words), dtype=int)
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
    sense_count[word_count] = len(unique_word_synsets)
    word_count = word_count + 1

# Sense similarity matrix
sense_similarity_matrix = np.array([[0 for i in range(len(synsets))] for j in range(len(synsets))], dtype=float)
for i in range(0, len(synsets)):
    for j in range(0, len(synsets)):
        similarity = synsets[i].wup_similarity(synsets[j])
        if similarity == None:
            similarity = 0
        sense_similarity_matrix[i][j] = similarity

# Strategy space
strategy_space = np.array([[0 for i in range(len(synsets))] for j in range(len(words))], dtype=float)

# Assigning a normal distribution to the strategy space
count = 0
senses_start_index = np.zeros(len(words), dtype=int)
for i in range(0, len(words)):
    for j in range(0, sense_count[i]):
        strategy_space[i][j+count] = 1 / sense_count[i]
    senses_start_index[i] = count
    count += sense_count[i]

# Replicator dynamics
number_of_iterations = 10
for i in range(0, number_of_iterations):
    for player in range(0, len(words)):
        player_payoff = 0
        strategy_payoff = np.zeros((sense_count[player], 1), dtype=float)
        sense_preference_player_temp = strategy_space[player][senses_start_index[player]:senses_start_index[player] + sense_count[player]]
        sense_preference_player = np.zeros((sense_count[player], 1))
        for index in range(sense_count[player]):
            sense_preference_player[index] = sense_preference_player_temp[index]
        for neighbour in range(0, len(words)):
            if player == neighbour:
                continue
            payoff_matrix = np.array(sense_similarity_matrix[senses_start_index[player]:senses_start_index[player]+sense_count[player], senses_start_index[neighbour]:senses_start_index[neighbour]+sense_count[neighbour]], dtype=float)
            sense_preference_neighbour_temp = strategy_space[neighbour][senses_start_index[neighbour]:senses_start_index[neighbour]+sense_count[neighbour]]
            sense_preference_neighbour = np.zeros((sense_count[neighbour], 1))
            for index in range(sense_count[neighbour]):
                sense_preference_neighbour[index] = sense_preference_neighbour_temp[index]
            current_payoff = np.array(word_similarity_matrix[player][neighbour] * np.dot(payoff_matrix, sense_preference_neighbour), dtype=float)
            strategy_payoff = np.add(current_payoff, strategy_payoff)
            player_payoff = np.dot(sense_preference_player.T, current_payoff) + player_payoff
        updation_values = np.ones(strategy_payoff.shape)
        if not player_payoff == 0:
            updation_values = np.divide(strategy_payoff, player_payoff)
        for j in range(0, sense_count[player]):
            strategy_space[player][senses_start_index[player]+j] = strategy_space[player][senses_start_index[player]+j] * updation_values[j]

# Displaying the obtained meanings
for word in range(0, len(words)):
    print(words[word] + ": ", end="")
    max_value = 0
    required_synset = None
    for synset in range(0, len(synsets)):
        if strategy_space[word][synset] > max_value:
            max_value = strategy_space[word][synset]
            required_synset = synsets[synset]
    print(required_synset.definition())

# Input: There is a financial institution near the river bank.
# Input: He went to the bank to deposit money.
