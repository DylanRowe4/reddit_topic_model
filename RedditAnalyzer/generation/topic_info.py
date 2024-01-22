import numpy as np
import networkx as nx
from openai import OpenAI
from nltk.cluster.util import cosine_distance

class TopicModelInfoGeneration:
    def __init__(self, openai_key):
        #instantiate openai model
        self.client = OpenAI(api_key = openai_key)

    def cosine_similarity(self, vector1, vector2):
        if np.isnan(1 - cosine_distance(vector1, vector2)):
            return 0
        return 1 - cosine_distance(vector1, vector2)

    def create_similarity_matrix(self, embeddings):
        #instantiate matrix of zeros (lenxlen)
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        #x values
        for x in range(len(embeddings)):
            #y values
            for y in range(len(embeddings)):
                #calculate similarity between two embeddings
                similarity = self.cosine_similarity(embeddings[x], embeddings[y])
                #leave diagonal of matrix (similarity to itself) as 0
                if x != y:
                    similarity_matrix[x][y] = similarity
        return similarity_matrix

    def rank_messages(self, topic_dataframe):
        #create copy of dataframe
        dataframe = topic_dataframe.copy()

        ranked_messages = []
        #iterate through all topics in dataframe
        for i in range(len(dataframe)):
            #save embeddings and text of topic
            embeddings = dataframe['embedding'][i]
            messages = dataframe['text'][i]

            #compute similarity matrix of each topics embeddings
            embedding_similarity_matrix = self.create_similarity_matrix(embeddings)

            #convert numpy array to graph
            embedding_similarity_graph = nx.from_numpy_array(embedding_similarity_matrix)

            #score each embedding using pagerank algorithm
            scores = nx.pagerank(embedding_similarity_graph)

            #rank messages in descending order based on importance (relative to other messages)
            ranked_scores = sorted(
                ((scores[idx], messages[idx]) for idx, s in enumerate(embeddings)), reverse=True)
            #append only text and remove score
            ranked_messages.append([tup[1] for tup in ranked_scores])
        dataframe['text'] = ranked_messages
        return dataframe
    
    def create_prompts(self, dataframe):           
        #prompt to create representative title
        dataframe['title_prompt'] = [
            f"""You are tasked with creating a readable and understandable topic title to represent the posts of users from a sub-reddit channel. 
            The title you create should be less than six words in length.

            For the following Reddit posts and key words; read and analyze each post and key word provided, think about the most important information,
            then create a topic title that represents the information in the posts and key words using the thought process below.

            Each Reddit posts is separated by the string "-----"

            POSTS: {"-----".join(post)[:13000]}
            TOPIC KEYWORDS: {key_words}

            Thought Process:
            1. Read and analyze the entire text, focusing on the most important parts of the posts.
            2. Answer the question "What is the main topic of discussion within these posts?" as concisely and completely as possible.
            3. Create a topic title that is only a few words long and will help readers understand the items discussed by the posts and keywords.
            4. Make sure your answers are accurate to the input text and do not make up information.
            5. Do not make up information or you will be penalized.
            6. You should only create one title.

            The output should be a short but concise title no longer than six words."""
            for post, key_words in zip(dataframe['text'], dataframe['key_words'])]

        #prompt to create summary of posts within topic
        dataframe['summary_prompt'] = [
            f"""
            You are tasked with reading the posts from within a sub-reddit channel and summarizing the information into an informative and descriptive
            summary. Your summary should represent the main topic of discussion by the users and their Reddit posts.

            For the following Reddit posts, read and analyze every post, think about the most important information then create an accurate and representative
            summary of the information.

            Each Reddit posts is separated by the string "-----"

            POSTS: {"-----".join(posts)[:13000]}

            To create the summary, please follow the following thought process.

            Thought Process:
            1. Read and analyze the entire text, focusing on the most important parts of the posts.
            2. Answer the question "What are the users within these posts discussing?" as concisely and completely as possible.
            3. Create a summary that will help readers understand the items discussed by the posts.
            4. The summary should include details regarding what the users are discussing in their posts.
            5. Make sure your answers are accurate to the input text and do not make up information.
            6. Do not make up information or you will be penalized.
            7. Do not repeat information.

            Your summary output should be no more than 4 sentences.""" for posts in dataframe['text']]
        return dataframe
    
    def generate_completion(self, prompt, model="gpt-3.5-turbo"):
        #send prompt to openai api
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }],
            temperature=0.4,
            top_p=1
        )
        output = response.choices[0].message.content
        return output
    
    def generate_titles_and_summaries(self, dataframe):
        #use similarity scores to rank sentence by importance incase we need to shorten prompt
        dataframe = self.rank_messages(dataframe)
        
        #create prompts for title and summary generation
        dataframe = self.create_prompts(dataframe)
        
        #generate topic titles
        dataframe['title'] = [self.generate_completion(prompt) for prompt in dataframe['title_prompt']]

        #generate topic summaries
        dataframe['summary'] = [self.generate_completion(prompt) for prompt in dataframe['summary_prompt']]
        return dataframe
