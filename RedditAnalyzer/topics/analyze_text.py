import pandas as pd
import numpy as np
import math, spacy
import umap, hdbscan, optuna
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from optuna.samplers import TPESampler
from IPython.display import clear_output

#silence optuna feedback from trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CreateTopicModel:
    def __init__(self, df, column_name, use_hdbscan=True):
        self.study_df = df.loc[df[column_name].apply(len) <= 1000].reset_index(drop=True).copy()
        self.column_name = column_name
        self.use_hdbscan = use_hdbscan
        
        ### Load Models Needed for BERTopic Analysis ###
        #instantiate keybert
        self.keybert = KeyBERTInspired()
        
        #load spacy model
        spacy_sm = spacy.load('en_core_web_sm')
        #format as part of speech model for bertopic
        self.pos = PartOfSpeech(spacy_sm)
        
        #instantiate mmr model for more diverse results and keywords
        self.mmr = MaximalMarginalRelevance(diversity=0.5)

        #build representation model pipeline for bertopic
        self.representation_model = {"KeyBERT": self.keybert, "MMR": self.mmr, "POS": self.pos}
        
        #load sentence transformer
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        #count vectorizer model for stop word removal
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def within_topic_similarity(self, topic_embeddings_dict):
        topic_sims = []
        for topic in topic_embeddings_dict.keys():
            #pairwise cosine similarity between topic keywords
            cos_sim = cosine_similarity(topic_embeddings_dict[topic])
            #set similarity to itself 0
            np.fill_diagonal(cos_sim, 0)

            #total similarity
            total_sim = np.sum(cos_sim)
            #number of comparisons
            count = cos_sim.shape[0] * cos_sim.shape[1]

            #add to list
            topic_sims.append(total_sim)
        #return number of topics and average within topic similarity
        return len(topic_sims), np.sum(topic_sims) / len(topic_sims)
    
    def between_topic_similarity(self, topic_embeddings_dict):
        topic_sims = []
        for topic_x in topic_embeddings_dict.keys():
            for topic_y in topic_embeddings_dict.keys():
                if topic_x == topic_y:
                    continue
                else:
                    cos_sim = cosine_similarity(topic_embeddings_dict[topic_x], topic_embeddings_dict[topic_y])

                    #total similarity
                    total_sim = np.sum(cos_sim)
                    #number of comparisons
                    count = cos_sim.shape[0] * cos_sim.shape[1]

                    #add to list
                    topic_sims.append(total_sim)
        #return number of topics compared and average between topic similarity
        return len(topic_sims), np.sum(topic_sims) / len(topic_sims)
    
    def total_similarity(self, df, embed_model, topk=15):
        #remove noise cluster from analysis
        df = df.loc[df['topic'] != -1].reset_index(drop=True)

        #combine all text under each topic
        docs_per_topic = df.groupby('topic').agg({self.column_name: ' '.join})

        #fit count vectorizer to post text
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        count_v = vectorizer.fit(df[self.column_name])

        #get features from cv
        features = vectorizer.get_feature_names_out()

        #also transform combined topic text
        X = count_v.transform(docs_per_topic[self.column_name])

        #fit within class tfidf algorithm from bertopic
        ctfidf_v = ClassTfidfTransformer(bm25_weighting=False, reduce_frequent_words=True)
        ctfidf = ctfidf_v.fit_transform(X)

        #pairwise cosine similarity(ctfidf)
        cosine_sim = cosine_similarity(ctfidf)

        topic_features = {}
        topic_feature_embeddings = {}
        #extract most important words from within each topics (ctfidf)
        for idx, topic in enumerate(ctfidf.toarray()):
            #sort topic features by importance
            top_features = topic.argsort()[::-1]

            #join them into one string
            keywords = [features[i] for i in top_features][:topk]
            topic_features[idx - 1] = keywords

            #create embeddings for top words
            topic_feature_embeddings[idx - 1] = self.embedding_model.encode(keywords)

        #calculate within topic similarity
        sims, within_topic_sim = self.within_topic_similarity(topic_feature_embeddings)

        if len(topic_feature_embeddings) >= 2:
            #calculate between topic similarity
            divs, between_topic_sim = self.between_topic_similarity(topic_feature_embeddings)
        else:
            divs, between_topic_sim = 0, 0

        #return within minus between topic similarity
        return within_topic_sim - between_topic_sim
    
    def optuna_gridsearch(self, trial):
        #parameter grid to select hyperparameters from
        n_neighbors = trial.suggest_int("n_neighbors", 2, 32, 5)
        n_components = trial.suggest_int("n_components", 2, 6)
        min_dist = trial.suggest_float("min_dist", 0.1, 1)
        metric = trial.suggest_categorical("metric", ['euclidean', 'cosine', 'manhattan'])
        
        if self.use_hdbscan:
            #parameters for hdbscan
            min_cluster_size = trial.suggest_int("min_cluster_size", 5, 100, 5)
            min_samples = trial.suggest_int("min_samples", 5, 100, 5)
            
            #clustering model
            clust_m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', prediction_data=True)
        else:
            #parameters for kmeans
            n_clusters = trial.suggest_int("n_clusters", 2, 20)
            
            #clustering model
            clust_m = KMeans(n_clusters=n_clusters)

        #dimensionality reduction model
        umap_m = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                           min_dist=min_dist, metric=metric, random_state=123)
        
        #instantiate bertopic with run of parameters
        model = BERTopic(umap_model=umap_m,
                         hdbscan_model=clust_m,
                         embedding_model=self.embedding_model,
                         vectorizer_model=self.vectorizer,
                         representation_model=self.representation_model)
        #fit bertopic with run of parameters
        topics, probs = model.fit_transform(self.study_df[self.column_name], self.embeddings)
        #save topics as column to dataframe
        self.study_df['topic'] = topics
        
        #calculate within and between topic similarities
        within_between = self.total_similarity(self.study_df, self.embedding_model)
        return within_between
    
    def fit_model(self, params, embeds=None):
        #fit bertopic using hyperparameters for umap and kmeans
        umap_m = umap.UMAP(n_neighbors=params['n_neighbors'], n_components=params['n_components'],
                           min_dist=params['min_dist'], metric=params['metric'], random_state=123)
        if self.use_hdbscan:
            clust_m = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'],
                                      metric='euclidean', prediction_data=True)
        else:
            clust_m = KMeans(n_clusters=params['n_clusters'])
        
        if embeds is None:
            #create embeddings for optuna study of BERTopic
            embeds = self.embedding_model.encode(self.study_df[self.column_name].tolist(), show_progress_bar=True)
            
        #instantiate bertopic
        model = BERTopic(umap_model=umap_m,
                         hdbscan_model=clust_m,
                         embedding_model=self.embedding_model,
                         vectorizer_model=self.vectorizer,
                         representation_model=self.representation_model)
        #fit bertopic with best parameters
        topics, probs = model.fit_transform(self.study_df[self.column_name], embeds)

        #assign topics to dataframe
        self.study_df['topic'] = topics
        self.study_df['embedding'] = [embed for embed in embeds]
        
        #groupby topic and create a list of all posts
        grouped_topics = self.study_df.groupby('topic', as_index=False).agg({'text': list, 'embedding': list})
        
        #get keywords for each topic
        grouped_topics['key_words'] = [
            [word[0] for word in model.get_topic(i)][:6] for i in range(len(model.generate_topic_labels()))
        ]
        return self.study_df, grouped_topics, model
    
    def find_hyperparams(self, num_trials):
        
        print('Creating embeddings...')
        #create embeddings for optuna study of BERTopic
        self.embeddings = self.embedding_model.encode(self.study_df[self.column_name].tolist(), show_progress_bar=True)
        
        #set seed for reproducability
        sampler = TPESampler(seed=123)
        #use optuna to maximize within similarity and minimize between similarity
        study = optuna.create_study(direction='maximize', sampler=sampler)
        print('Tuning parameters...')
        #optimize results using custom metric
        study.optimize(self.optuna_gridsearch, n_trials=num_trials, show_progress_bar=True)
        #clear progress bar and text
        clear_output()
        
        #fit bertopic model using hyperparameters
        optimal_df, grouped_topics, optimal_model = self.fit_model(study.best_params.copy(), self.embeddings)
        return optimal_df, grouped_topics, optimal_model
