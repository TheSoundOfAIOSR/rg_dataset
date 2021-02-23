import praw
import json

SAVED_RESULTS = ['reddit.jsonl', 'reddit.json']

# you can use your own credentials if you want
CLIENT_ID = 'EZ57p7KGDL0_ZQ'
CLIENT_SECRET = 'FXYfQdTXOpiHEnF94s1Kcg2oeHeqWA'
USER_AGENT = 'Reddit WebScrapping'



def authenticate(id, secret, name):
    ''' authentication function to the Reddit instance
    To create your own Reddit instance follow this link
    https://www.reddit.com/prefs/apps

    :param client_id: The Reddit instance ID found in the top left
    :param client_secret: The Reddit instance secret token
    :param user_agent: The Reddit instance name
    :return: the Reddit instance created
    '''

    reddit = praw.Reddit(client_id=id, client_secret=secret, user_agent=name)
    return reddit

def scraping_submissions(sub="all", search_tag="guitar timbre", file_path=SAVED_RESULTS, json_lines = True):
    ''' function that search the tag in all the subreddits, then loops through all the submissions and store them and
        their comments in a dictionary.

    :param sub: use 'all' if you want to search in all the subreddits, otherwise use its specific name. E.g. RoastMe
    :param search_tag: the tag you want to look for in the subreddits / submissions
    :param file_path: list containing all possible paths to store data in
    :param json_lines: True if you want the output in a JSON Lines format, False if you want the normal JSON format
    :return: a dictionary containing (submission : comments) pairs
    '''

    instance = authenticate(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    sub_comments = {}

    # loop through all the submissions and extract their comments

    # search the tag guitar timbre through all subreddits
    for submission in instance.subreddit(sub).search(search_tag):

        submission.comments.replace_more(limit=0)
        comments = []

        for top_level_comment in submission.comments:

            comments.append(top_level_comment.body)

        # store the comments in a dict with key(submission) and value(list of comments of the submission)
        sub_comments[submission.title] = comments

    # dumb into a json file
    if json_lines == True:
        with open(file_path[0], 'w') as fp:
            json.dump(sub_comments, fp, indent=4)
    else:
        with open(file_path[1], 'w') as fp:
            json.dump(sub_comments, fp, indent=4)

if __name__ == "__main__":

    # authenticate to the reddit instance created
    #instance = authenticate(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    # scraping comments from Reddit
    scraping_submissions("all", "guitar timbre", SAVED_RESULTS, True)