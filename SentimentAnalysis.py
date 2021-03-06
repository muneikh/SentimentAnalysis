import fasttext

# train the classifier from scratch
# classifier = fasttext.supervised('train.txt', 'model', label_prefix='__label__')

# if the model already exists no need to train it again
classifier = fasttext.load_model('model.bin', label_prefix='__label__')

# result = classifier.test('test.txt')
# print 'P@1:', result.precision
# print 'R@1:', result.recall
# print 'Number of examples:', result.nexamples

texts = ['should I go to school now?', 'I hate you']

# labels = classifier.predict(texts, k=3)
# print labels

# Or with the probability
labels = classifier.predict_proba(texts, k=3)
print labels