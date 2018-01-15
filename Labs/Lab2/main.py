import lab2
import utils

#-------------------------------------------------------------------------------
# Data Loading
#-------------------------------------------------------------------------------

train_data = utils.load_reviews_data('../../Data/reviews_train.csv')
val_data = utils.load_reviews_data('../../Data/reviews_val.csv')
test_data = utils.load_reviews_data('../../Data/reviews_test.csv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = lab2.bag_of_words(train_texts)

features_train = lab2.Features(train_texts, dictionary)
features_val = lab2.Features(val_texts, dictionary, train_texts)
features_test = lab2.Features(test_texts, dictionary, train_texts)

train_bow_features = features_train.extract()
val_bow_features = features_val.extract()
test_bow_features = features_test.extract()

# You may modify the following when adding additional features (Part 3c)

train_stopword_features = features_train.remove_stop_words().extract()
val_stopword_features = features_val.remove_stop_words().extract()
test_stopword_features = features_test.remove_stop_words().extract()

features_train.show_most_useful_n_features()

train_uncommon_features = features_train.reset().remove_uncommon().extract()
val_uncommon_features = features_val.reset().extract()
test_uncommon_features = features_test.reset().extract()


#-------------------------------------------------------------------------------
# Part 1 - Perceptron Algorithm
#-------------------------------------------------------------------------------

# toy_features, toy_labels = utils.load_toy_data('../../Data/toy_data.csv')
#
# theta, theta_0 = lab2.perceptron(toy_features, toy_labels, T=5)
#
# utils.plot_toy_results(toy_features, toy_labels, theta, theta_0)

#-------------------------------------------------------------------------------
# Part 2 - Classifying Reviews
#-------------------------------------------------------------------------------

theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T=5)

train_accuracy = lab2.accuracy(train_bow_features, train_labels, theta, theta_0)
val_accuracy = lab2.accuracy(val_bow_features, val_labels, theta, theta_0)
#
# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))

#-------------------------------------------------------------------------------
# Part 3 - Improving the Model
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Tuning the Hyperparameters
#-------------------------------------------------------------------------------

# Ts = [1, 5, 10, 15, 20]

# Ts = [i for i in range(5, 101, 5)]
#
# train_accs, val_accs = lab2.tune(Ts, train_bow_features, train_labels, val_bow_features, val_labels)
#
# utils.plot_tune_results(Ts, train_accs, val_accs)

#-------------------------------------------------------------------------------
# Best T value
#-------------------------------------------------------------------------------

T_best = 20  # You may modify this value

#-------------------------------------------------------------------------------
# Part 3.2 - Understanding the Model
#-------------------------------------------------------------------------------

theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T_best)

word_list = sorted(dictionary.keys(), key=lambda word: dictionary[word])
sorted_words = utils.most_explanatory_words(theta, word_list)

print("Top 100 most explanatory words")
print(sorted_words[:100])

#-------------------------------------------------------------------------------
# Part 3.3 - Adding Features
#-------------------------------------------------------------------------------

theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T_best)

train_accuracy = lab2.accuracy(train_bow_features, train_labels, theta, theta_0)
val_accuracy = lab2.accuracy(val_bow_features, val_labels, theta, theta_0)

print("Bag-of-words features")
print("Training accuracy: {:.4f}".format(train_accuracy))
print("Validation accuracy: {:.4f}".format(val_accuracy))

print()

theta, theta_0 = lab2.perceptron(train_stopword_features, train_labels, T_best)

train_accuracy = lab2.accuracy(train_stopword_features, train_labels, theta, theta_0)
val_accuracy = lab2.accuracy(val_stopword_features, val_labels, theta, theta_0)

print("Without stopwords features")
print("Training accuracy: {:.4f}".format(train_accuracy))
print("Validation accuracy: {:.4f}".format(val_accuracy))

print()

theta, theta_0 = lab2.perceptron(train_uncommon_features, train_labels, T_best)

train_accuracy = lab2.accuracy(train_uncommon_features, train_labels, theta, theta_0)
val_accuracy = lab2.accuracy(val_uncommon_features, val_labels, theta, theta_0)

print("Without uncommon features")
print("Training accuracy: {:.4f}".format(train_accuracy))
print("Validation accuracy: {:.4f}".format(val_accuracy))

#-------------------------------------------------------------------------------
# Part 4 - Testing the Model
#-------------------------------------------------------------------------------

# theta, theta_0 = lab2.perceptron(train_final_features, train_labels, T_best)

# train_accuracy = lab2.accuracy(train_final_features, train_labels, theta, theta_0)
# val_accuracy = lab2.accuracy(val_final_features, val_labels, theta, theta_0)
# test_accuracy = lab2.accuracy(test_final_features, test_labels, theta, theta_0)

# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))
# print("Test accuracy: {:.4f}".format(test_accuracy))
