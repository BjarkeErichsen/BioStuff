from bs4 import BeautifulSoup
import requests

html_text = requests.get('https://www.crummy.com/software/BeautifulSoup/bs4/doc/').text #we read the website html, and get only text
soup = BeautifulSoup(html_text, 'lxml')              #getting the html code into a usefull format
a_section = soup.find('head')  #finding all html blocks of a specific type  #, class_='card'
a_section.find('p').child




