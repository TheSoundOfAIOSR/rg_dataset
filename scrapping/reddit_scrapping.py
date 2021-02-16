import praw
import json

SAVED_RESULTS = 'reddit.json'

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

def scrapping_submissions(subreddit="all", search_tag="guitar timbre", file_path=SAVED_RESULTS):
    ''' function that search the tag in all the subreddits, then loops through all the submissions and store them and
        their comments in a dictionary.

    :param subreddit: use 'all' if you want to search in all the subreddits, otherwise use its specific name. E.g. RoastMe
    :param search_tag: the tag you want to look for in the subreddits / submissions
    :param file_path: the path and name of the JSON file that will contain our scrapped comments
    :return: a dictionary containing (submission : comments) pairs
    '''

    instance = authenticate(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    sub_comments = {}

    # loop through all the submissions and extract their comments

    # search the tag guitar timbre through all subreddits
    for submission in instance.subreddit("all").search("guitar timbre"):

        submission.comments.replace_more(limit=0)
        comments = []

        for top_level_comment in submission.comments:

            comments.append(top_level_comment.body)

        # store the comments in a dict with key(submission) and value(list of comments of the submission)
        sub_comments[submission.title] = comments

    # dumb into a json file

    with open(file_path, 'w') as fp:
        json.dump(sub_comments, fp, indent=4)

if __name__ == "__main__":

    # authenticate to the reddit instance created
    instance = authenticate(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    # test search option
    scrapping_submissions("all", "guitar timbre", SAVED_RESULTS)