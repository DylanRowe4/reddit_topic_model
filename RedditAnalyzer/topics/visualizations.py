import numpy as np
import pandas as pd
import string, re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from PIL import Image

#set matplotlib style
plt.style.use("dark_background")

stop_words = stopwords.words('english') + ['like']

def clean_text(sentence):
    #replace multiple spaces, tabs, new lines, etc
    sentence = re.sub('\s{2,}', ' ', sentence)
    #remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    #split tokens
    sentence = sentence.lower().split()
    #remove stop words
    sentence = [word for word in sentence if word not in stop_words]
    return sentence

def prepare_data(corpus):
    #clean text variable
    corpus = [clean_text(text) for text in corpus]
    corpus = [' '.join(text) for text in corpus if text != []]
    corpus = pd.DataFrame(corpus, columns=['text_tokens'])
    corpus['text_tokens'] = corpus['text_tokens'].apply(lambda txt: txt.split())
    return corpus

def bar_chart(ax, data, title, n_words=8):
    data = prepare_data(data)
    #explode series of tokens and get value count of words
    word_counts = data['text_tokens'].explode().value_counts()[:n_words]
    #plot bar graph
    word_counts.plot(kind='bar', color='#1a3258', edgecolor='white', ax=ax)
    ax.set_xticklabels(word_counts.index, rotation=45)
    ax.set_title(title)

    for idx, val in enumerate(word_counts):
        ax.text(idx, val, f"{val:,}", ha='center')

def generate_wordcloud(ax, data, title, mask=None):
    data = prepare_data(data)
    #create one corpus of all text
    data_corpus = ' '.join(data['text_tokens'].explode())
    #create wordcloud
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap='RdYlGn',
                      mask=255 - mask,
                      background_color='black',
                      stopwords=stop_words,
                      collocations=True,
                      contour_color='#5d0f24',
                      contour_width=3).generate_from_text(data_corpus)
    ax.imshow(cloud)
    ax.axis('off')
    ax.set_title(title)
    
def plot_bar_charts(df1, title1):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    #bra graphs to compare posts and comments
    bar_chart(ax1, df1, title1, 10)
    ax1.set_title(title1)
    plt.show()

def plot_wordclouds(df1, title1):
    #import logo as mask
    mask = Image.open('./RedditAnalyzer/RedditLogo.png')
    mask = np.array(mask)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    #use function to generate wordcloud
    generate_wordcloud(ax1, df1, title1, mask=mask)
    ax1.set_title(title1)
    plt.show()
    
def text_plots(df1, title1, title2, time_frame='This Week'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    #import logo as mask
    mask = Image.open('./RedditAnalyzer/RedditLogo.png')
    mask = np.array(mask)
    
    #bra graphs to compare posts and comments
    bar_chart(ax1, df1, title1, 10)
    ax1.set_title(title1)
    
    #use function to generate wordcloud
    generate_wordcloud(ax2, df1, title2, mask=mask)
    ax2.set_title(title2)
    fig.suptitle(time_frame)
    plt.show()