import requests
from bs4 import BeautifulSoup
import re

PAGE = str(2)
OTHER_PAGE = '?page='

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

    print("Done fetching.")

    # remove redundant URLs that get scraped
    unique_threads = set(threads)

    return  unique_threads


def max_pages(url=BASE_URL):
    ''' find the max number of pages to loop through

    :param url: URL of specific thread
    :return: max page
    '''
    URL = url
    page = fetch_page(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    num_pages =  soup.find('span', {'class': 'pageNavHeader'})

    return num_pages





if __name__ == "__main__" :
    print('ok')
    #threads = fetch_threads()
    max_page = max_pages('https://www.tdpri.com/posts/10377555/')
    print(type(max_page))
    #print(type(max_page.string))
    #print(max_page.string[-1])