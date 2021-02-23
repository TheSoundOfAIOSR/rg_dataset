import requests
from bs4 import BeautifulSoup
import re

PAGE = str(2)
OTHER_PAGE = '?page='

#BASE_URL = 'https://www.tdpri.com/search/'
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
        # each result page contains
        if counter <= 25 :
            url = url_root.format(empty_str, empty_str)
            page = fetch_page(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            print("Fetching first batch of threads...")
            for link in soup.find_all('a', href=re.compile('posts/')):

                threads.append(link.get('href'))
                counter += 1

        else :
            url = url_root.format(other_page, i)
            page = fetch_page(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            print("Fetching batch n°{}...".format(i))
            for link in soup.find_all('a', href=re.compile('posts/')):

                threads.append(link.get('href'))
                counter += 1
    print("Done fetching.")

    unique_threads = set(threads)
    return  unique_threads



if __name__ == "__main__" :
    threads = fetch_threads()