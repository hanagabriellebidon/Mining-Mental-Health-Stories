import nltk
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import json
import re

import matplotlib.pyplot as plt
import seaborn as sns

mental_health_story_1 = open('hanagabriellebidon.txt', 'r').read()
sentences = sent_tokenize(mental_health_story_1)
sentences_dict = {}
for index in range(len(sentences)):
    element = sentences[index]
    key = 'sentence_' + str(index)
    sentences_dict[key] = element
sentences_df = pd.DataFrame.from_dict(sentences_dict, orient='index')
print(sentences_df)

# https://github.com/prateekjoshi565/topic_modeling_online_reviews/blob/master/Mining_Online_Reviews_using_Topic_Modeling_%28LDA%29.ipynb
# Get the 20 most frequent words in the mental health story
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

frequency_of_words = freq_words(sentences_df[0])
