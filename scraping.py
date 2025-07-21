data = []
headers = ["id", "address", "bedroom_nums", "bathroom_nums", "car_spaces", "land_size", "house_type", "price"]

data.append(headers)

cnt = 0

import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By


for step in range(0, 8):
    driver = uc.Chrome()
    for i in range(step * 9  + 1 , step * 9 + 10):
        cnt += 1
        driver.get(f"https://www.realestate.com.au/sold/property-house-in-nsw/list-{i}")
        time.sleep(3)
        listings = driver.find_elements(By.CSS_SELECTOR, 'a.details-link')
        urls = [l.get_attribute('href') for l in listings]
        print(cnt)
        for url in urls:
            bedroom_nums = None
            bathroom_nums = None
            car_spaces = None
            land_size = None
            house_type = None
            driver.get(url)
            time.sleep(0.1)
            try:
                address = driver.find_element(By.CSS_SELECTOR, 'h1.property-info-address').text
            except:
                address = None
            try:
                try:
                    house_type = driver.find_element(By.CSS_SELECTOR, 'ul.styles__Wrapper-sc-xhfhyt-1.bGRFcz.property-info__primary-features').text.split(' ')[0].split('\n')[-1]
                except:
                    house_type = None
                items = driver.find_elements(By.XPATH, "//ul[contains(@class, 'property-info__primary-features')]//li[@aria-label]")
                for item in items:
                    if 'bedroom' in item.get_attribute("aria-label"):
                        bedroom_nums = item.get_attribute("aria-label").split(' ')[0]
                    if 'bathroom' in item.get_attribute("aria-label"):
                        bathroom_nums = item.get_attribute("aria-label").split(' ')[0]
                    if 'car' in item.get_attribute("aria-label"):
                        car_spaces = item.get_attribute("aria-label").split(' ')[0]
                    if 'size' in item.get_attribute("aria-label"):
                        land_size = item.get_attribute("aria-label").split(' ')[0]
                    # print(item.get_attribute("aria-label"))
        
            except:
                print(2)
            try:
                price = driver.find_element(By.CSS_SELECTOR, 'span.property-price.property-info__price').text
            except:
                price = None
            data.append([cnt, address, bedroom_nums, bathroom_nums, car_spaces, land_size, house_type, price])
            # print(price)
    
    # print(data)
    driver.quit()

import pandas as pd

# assuming your 2D list is called data, with first row as header
df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv('rawdataset.csv', index=False)
