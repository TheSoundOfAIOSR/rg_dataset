import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm

PAGE = str(2)
OTHER_PAGE = '?page='
TDPRI = 'tdpri.json'

BASE_URL = 'https://www.tdpri.com/'
#QUERY_PARAM = '97606591/{}{}?q=timbre&o=relevance'
URL_ROOT = 'https://www.tdpri.com/search/97606591/{}{}?q=timbre&o=relevance'
NUM_PAGES = 20

# create BeautifulSoup object


def fetch_page(url = URL_ROOT):
    ''' perform an HTTP request to the given URL (base + query) and
    returns the HTML data.

    :param url: base url of the website / forum you want to scrape
    :param query: query to search for a specific word
    :return: returns Python object containing the information found
    '''

    URL = url

    page = requests.get(URL)

    return page

def fetch_threads(n_pages=NUM_PAGES, url_root = URL_ROOT, other_page = OTHER_PAGE ):
    ''' store all the URLs related to a thread found in a page

    :param n_pages: number of pages to loop through (E.g. each page in TDRPI contains 25 threads)
    :param url_root: URL that sould contains the threads
    :param other_page: query to add to url_root to fetch other pages (
    keep in mind that the 1st result page in TDRPI has a url query as follows : /search/97606591/?q=timbre&o=relevance
    while the other pages in TDRPI have a url query as follows : /search/97606591/?page=2&q=timbre&o=relevance
    :return: set of links of each thread in the search result
    '''

    threads = []
    counter = 0
    empty_str = ''
    
    for i in range(1, n_pages+1) :
        # each result page contains 25 threads
        # the 1st result page has a different URL, thus needs to be separated
        if counter <= 25 :
            # insert empty str to the beginning of the query to get the 1st page URL
            url = url_root.format(empty_str, empty_str)
            page = fetch_page(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            print("Fetching first batch of threads...")
            # fetch all the tags for a link similar to 'post/'
            # each link has an href attribute as follows 'posts/int'
            for link in soup.find_all('a', href=re.compile('posts/')):
                # append to list all the links of relative threads
                threads.append(link.get('href'))
                counter += 1

        else :
            # insert ?page= and page number to the beginning of the query to get the other pages URL
            url = url_root.format(other_page, i)
            page = fetch_page(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            print("Fetching batch n°{}...".format(i))
            for link in soup.find_all('a', href=re.compile('posts/')):
                threads.append(link.get('href'))
                counter += 1

    print("Done fetching.\n")

    # remove redundant URLs that get scraped
    unique_threads = set(threads)

    return  unique_threads


def max_pages(url=BASE_URL):
    ''' find the maximum number of pages to loop through

    :param url: URL of specific thread
    :return: max page
    '''

    URL = url
    page = fetch_page(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    n_page = 0

    # search for specific tag that contains the navbar to move from one page to another
    num_pages = soup.find('span', {'class': 'pageNavHeader'})

    # Condition to avoid the AttributeError exception if a thread has only one page
    if not num_pages :
        n_page = 1
    else :
        n_page = int(num_pages.string[-2:])

    return n_page


def data_scraping(base_url = BASE_URL):
    ''' scrape data from TDPRI website.

    :param base_url: base root URL https://www.tdpri.com/ to which we will add each thread relative path
    :return: dictionary containing the thread title and all its comments
    '''

    thread_comments = {}

    # fetch all thread's urls
    threads = fetch_threads()

    # loop through all thread links
    for relative_link in tqdm(threads):

        thread_url = base_url + relative_link

        # for each thread, find the max number of comment pages it has
        n_pages = max_pages(thread_url)

        comments = []

        # threads with only one page have a different URL
        if n_pages == 1 :
            page = fetch_page(thread_url)
            soup = BeautifulSoup(page.content, 'html.parser')

            title = soup.find('h1')

            for comment in soup.find_all('blockquote', class_ = re.compile('messageText SelectQuoteContainer ugc baseHtml')):
                comments.append(comment.text)
            thread_comments[title.text] = comments

        else :
            page_tmp = fetch_page(thread_url)
            soup_tmp = BeautifulSoup(page_tmp.content, 'html.parser')
            title = soup_tmp.find('h1')
            store_page = ''
            for i in range(1, n_pages + 1) :

                # first page of comments have a different URL than other pages
                # E.g. https://www.tdpri.com/threads/is-tone-really-that-important.708613/
                # while 2nd to last pages have the suffix "page-x" as shown here
                # https://www.tdpri.com/threads/is-tone-really-that-important.708613/page-2
                if i == 1:
                    page_url = soup_tmp.find('a', href=True, text=str(i)).get('href')
                    store_page = page_url
                    comment_url = 'https://www.tdpri.com/' + page_url
                    page = fetch_page(comment_url)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    for comment in soup.find_all('blockquote',
                                                 class_=re.compile('messageText SelectQuoteContainer ugc baseHtml')):
                        comments.append(comment.text)

                else :
                    comment_url = 'https://www.tdpri.com/' + store_page + 'page-'+str(i)
                    page = fetch_page(comment_url)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    for comment in soup.find_all('blockquote',
                                                 class_=re.compile('messageText SelectQuoteContainer ugc baseHtml')):
                        comments.append(comment.text)
            # store (thread title, thread comments) in a dictionary
            thread_comments[title.text] = comments

    return thread_comments


def save_json(threads, filename = TDPRI):
    ''' dump dictionaries in JSON files

    :param threads: dictionary
    :param filename: name of the file to store the dictionary into
    :return:
    '''
    with open(filename, 'w') as fp:
        print("Saving scraped data to JSON file...")
        json.dump(threads, fp, indent=4)
        print("Done !")

if __name__ == "__main__" :

    threads = data_scraping()
    save_json(threads)


