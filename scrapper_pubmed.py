# Import necessary libraries and modules from selenium, tqdm, and standard Python libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm.notebook import tqdm  # For displaying progress bars in Jupyter Notebooks
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time  # For adding delays
import os  # For interacting with the operating system
import glob  # For file path matching

# Define the Scrapper class for scraping data from PubMed
class Scrapper:
    def __init__(self, webdriver_path='/usr/local/bin/chromedriver', headless=True):
        # Initialize the Scrapper with default values for the webdriver path, download directory, and headless mode
        
        self.webdriver_path = webdriver_path  # Path to the Chrome WebDriver executable
        self.headless = headless  # Whether to run the browser in headless mode (without UI)
        self.driver = None  # WebDriver instance, initially set to None

    # Method to set up the Chrome WebDriver with the specified options
    def setup_driver(self):
        # Create an instance of Chrome options
        chrome_options = Options()
        if self.headless:
            # Add headless mode if specified
            chrome_options.add_argument('--headless=new')
        # Initialize the WebDriver with the specified options and executable path
        self.driver = webdriver.Chrome(service=Service(executable_path=self.webdriver_path), options=chrome_options)
    
    # Method to close the WebDriver
    def close(self):
        if self.driver:
            # Quit the WebDriver if it has been initialized
            self.driver.quit()

    # Method to start the scraping process
    def start(self, citations):
        # Initialize lists to store the scraped titles, abstracts, DOIs, and publication types
        titles = []
        abstracts = []
        doi = []
        pub_type = []
        
        # Set up the WebDriver
        self.setup_driver()
        time.sleep(1)  # Small delay to ensure the WebDriver is fully initialized
        
        # Navigate to the PubMed website
        self.driver.get('https://pubmed.ncbi.nlm.nih.gov/')
        time.sleep(3)  # Allow the page to load completely

        # Loop through each citation in the provided list
        for paper in tqdm(citations, desc="Scraping Information"):
            
            # Locate the search text box on the PubMed homepage and enter the citation
            text_box = self.driver.find_element(By.XPATH, '/html/body/div[2]/main/div[1]/div/form/div/div[1]/div/span/input')
            text_box.clear()  # Clear any existing text in the search box
            text_box.send_keys(paper)  # Input the citation into the search box
            
            # Locate and click the search button
            search_button = self.driver.find_element(By.XPATH, '/html/body/div[2]/main/div[1]/div/form/div/div[1]/div/button')
            search_button.click()
            time.sleep(1)  # Wait for the search results to load

            # Try to extract the title of the paper
            try:
                titles.append(self.driver.find_element(By.XPATH, '/html/body/div[5]/main/header/div[1]/h1').text)
            except NoSuchElementException:
                titles.append("Not Found")  # Append "Not Found" if the title is not available

            # Try to extract the DOI of the paper
            try: 
                doi.append(self.driver.find_element(By.XPATH,'/html/body/div[5]/main/header/div[1]/div[1]/span').text)
            except NoSuchElementException:
                doi.append("Not Found")  # Append "Not Found" if the DOI is not available

            # Try to extract the abstract of the paper
            try: 
                abstracts.append(self.driver.find_element(By.XPATH,'/html/body/div[5]/main/div[3]').text)
            except NoSuchElementException:
                abstracts.append("Not Found")  # Append "Not Found" if the abstract is not available
                
            # Optional: Try to extract the publication type of the paper (commented out)
            # try:
            #     pub_type.append(self.driver.find_element(By.XPATH, '/html/body/div[5]/main/header/div[1]/div[1]/div[1]').text)
            # except NoSuchElementException:
            #     pub_type.append("Not Found")

            # Navigate back to the search page to process the next citation
            self.driver.back()

        # Close the WebDriver after scraping is complete
        self.close()

        # Return the collected data: titles, abstracts, and DOIs
        return titles, abstracts, doi  # pub_type is not returned since it is commented out
