import pandas as pd
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

# Output data structure
data = []
headers = ["id", "address", "bedroom_nums", "bathroom_nums", "car_spaces", "land_size", "price"]
data.append(headers)

cnt = 0

# Create driver once



for step in range(0, 6):  # House size buckets
    driver = uc.Chrome()
    house_size_min = 200 + step * 200
    house_size_max = 200 + (step + 1) * 200
    print(f"Scraping size: {house_size_min}-{house_size_max} m²")

    for i in range(1, 80):  # Pages 1 to 80
        url = f"https://www.realestate.com.au/sold/property-house-size-{house_size_min}-{house_size_max}-in-nsw/list-{i}"
        #print(f"   Page {i}: {url}")
        driver.get(url)
        time.sleep(0.1)

        # Each card
        property_cards = driver.find_elements(By.CLASS_NAME, "residential-card__content")

        for card in property_cards:
            cnt += 1

            # Get address
            try:
                address = card.find_element(By.CLASS_NAME, "residential-card__details-link").text.strip()
            except:
                address = None

            # Get features
            bedroom_nums = bathroom_nums = car_spaces = land_size = price = None

            features = card.find_elements(By.XPATH, ".//ul[contains(@class, 'residential-card__primary')]//li[@aria-label]")
            for item in features:
                label = item.get_attribute("aria-label").lower()
                value = label.split(" ")[0]
                if "bedroom" in label:
                    bedroom_nums = value
                elif "bathroom" in label:
                    bathroom_nums = value
                elif "car" in label:
                    car_spaces = value
                elif "size" in label:
                    land_size = value


            # Get price
            try:
                price = card.find_element(By.CLASS_NAME, "property-price").text.strip()
            except:
                price = None

            # Add to dataset
            data.append([cnt, address, bedroom_nums, bathroom_nums, car_spaces, land_size, price])

        time.sleep(0.1)
    driver.quit()
    
# Save data to CSV
df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv("raw_dataset.csv", index=False)
print("✅ Data saved to nsw_sold_properties_cleaned.csv")