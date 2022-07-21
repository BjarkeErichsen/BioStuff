from bs4 import BeautifulSoup
import requests

html_text = requests.get('https://www.boligsiden.dk/tilsalg/kort').text #we read the website html, and get only text
soup = BeautifulSoup(html_text, 'lxml')              #getting the html code into a usefull format
course_card = soup.find_all('div')  #finding all html blocks of a specific type  #, class_='card'

soup.find_all()