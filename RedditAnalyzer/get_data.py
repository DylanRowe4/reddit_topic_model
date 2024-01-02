import praw
import time
import pandas as pd
from collections import defaultdict
from praw.models import MoreComments
# https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        try:
            #connect to reddit instance using credentials
            self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        except:
            print('Credentials are not valid.')
            
    #function to extract info from comments
    def extract_comments(self, comment):
        return {'text': comment.body, 'uid': comment.id, 'score': comment.score,
                'pid': comment.parent_id, 'created': comment.created_utc, 'parent': 0}

    #function to extract all info we want from post
    def extract_info(self, post, with_comments=True):
        #extract post information
        post_list =  [{'text': f"{post.title}. {post.selftext}".strip(), 'uid': post.id, 'score': post.score,
                  'pid': post.id, 'created': post.created_utc, 'parent': 1}]
        if with_comments:
            #sort post comments by new
            post.comment_sort = 'new'
            #remove all MoreComments objects from the forest
            post.comments.replace_more(limit=None)
            
            #retrieve comment info
            comments = [self.extract_comments(comment) for comment in post.comments.list()]
            time.sleep(1)
        else:
            comments = []
        return post_list + comments

    #select specific subreddit
    def subreddit_posts(self, subreddit_name, time_filt='week', with_comments=True):
        #create subreddit object (two subreddits = DuelLinks+MasterDuel)
        subreddit = self.reddit.subreddit(subreddit_name)

        #name of the subreddit
        print("Display Name:", subreddit.display_name)
        #title of the subreddit
        print("Title:", subreddit.title)
        #description of the subreddit
        #print("Description:", subreddit.description)

        #top posts of current month
        posts = subreddit.top(time_filter=time_filt, limit=None)

        #list comprehension to grab info from each post
        posts_cleaned = []
        counter = 0
        for post in posts:
            posts_cleaned += self.extract_info(post, with_comments)
            counter += 1

        #store in pandas dataframe
        posts = pd.DataFrame(posts_cleaned)
        print(f"Posts: {counter:,} | Comments: {len(posts) - counter:,}")
        return posts