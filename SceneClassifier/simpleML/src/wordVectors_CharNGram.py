
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import Word2Vec
from gensim.models import fasttext


# In[2]:


sentences = [] # list to store ngrams of all the sentences
#function to get ngrams. 
def sent2ngrams_simple(text, n=3):  # n =3 so, it returns character level trigrams
    x = [text[i:i+n] for i in range(len(text)-n+1) if not " " in text[i:i+n]]
    sentences.append(x)
    snt = ' '.join(x) 
    return snt


# In[3]:


sent2ngrams_simple('hello how are you')


# In[4]:


#load the data into a dataframe
#df = pd.read_excel('data.xlsx')


# In[5]:


#df['text'] has the text data
#temp = df['text'].map(lambda x: sent2ngrams_simple(x)) 


# In[ ]:


# train model
model = Word2Vec(sentences,min_count=1,size=25) # model = FastText(sentences, min_count=1,size=25)
# summarize the loaded model
print(model)
# summarize vocabulary
words = sorted(list(model.wv.vocab))
print(words)
# access vector for one word
#print(model['word'])

#print(model.most_similar('word'))


# In[ ]:


# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin') #new_model = FastText.load('model.bin')
print(new_model)

