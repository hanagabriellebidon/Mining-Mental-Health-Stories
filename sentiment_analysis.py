import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

mental_health_story = open('mentalhealth_hana_gabrielle.txt', 'r').read()
sentences = sent_tokenize(mental_health_story)

sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
#     for k in ss:
#         print(‘{0}: {1}, ‘.format(k, ss[k]), end=’’)
#     print()

# Remove punctuation from sentences.
# def remove_punctuation(sentence):
#     sentence = re.sub(r'[^\w\s]','', sentence)
#     return sentence
# cleaned_sent = [remove_punctuation(sentence) for sentence in sentences]
# partial_story = cleaned_sent
#
# partial_story_words = [word_tokenize(sentence) for sentence in partial_story]
# print(partial_story_words)
#
# stop_words = list(set(stopwords.words('english')))
# print('Number of stopwords:', len(stop_words))
# print(f'Stop words:\n{stop_words}')
#
# def remove_stopword(sentence):
#     return [w for w in sentence if not w in stop_words]
#
# filtered = [remove_stopword(s) for s in partial_story_words]
# print(filtered)
#
# POS = [nltk.pos_tag(tokenized_sent) for tokenized_sent in filtered]
# print(POS)
