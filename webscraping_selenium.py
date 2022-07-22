from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
#depreciated
#PATH_driver = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\trashfolder\chromedriver.exe"
#driver = webdriver.Chrome(PATH_driver)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

PATH_website = "https://www.techwithtim.net/"
driver.get(PATH_website)


from selenium.webdriver.common.keys import Keys
search = driver.find_element(by=By.CLASS_NAME, value= "search-field")
search.send_keys("gamers")    #inputting "gamers" into the searchbar
search.send_keys(Keys.RETURN) #pressing "enter"

time.sleep(5)
print(driver.title)
