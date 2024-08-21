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

# Define the SciHubDownloader class for downloading research papers from Sci-Hub
class SciHubDownloader:
    def __init__(self, webdriver_path='/usr/local/bin/chromedriver', download_directory="/Users/samyaksheth/Desktop/github/pdfs", headless=True):
        # Initialize the downloader with default values for the webdriver path, download directory, and headless mode
        
        self.webdriver_path = webdriver_path  # Path to the Chrome WebDriver executable
        self.download_directory = download_directory  # Directory where PDF files will be downloaded
        self.headless = headless  # Whether to run the browser in headless mode (without UI)
        self.driver = None  # WebDriver instance, initially set to None
        self.downloaded_files = []  # List to store the names of downloaded files

    # Method to set up the Chrome WebDriver with the specified options
    def setup_driver(self):
        # Create an instance of Chrome options
        chrome_options = Options()
        if self.headless:
            # Add headless mode if specified
            chrome_options.add_argument('--headless=new')

        if self.download_directory:
            # Set the default download directory if provided
            prefs = {"download.default_directory": self.download_directory}
            chrome_options.add_experimental_option("prefs", prefs)

        # Initialize the WebDriver with the specified options and executable path
        self.driver = webdriver.Chrome(service=Service(executable_path=self.webdriver_path), options=chrome_options)

    # Method to download papers using DOIs
    def download_paper(self, dois):
        # Set up the WebDriver
        self.setup_driver()
        time.sleep(2)  # Small delay to ensure the WebDriver is fully initialized
        
        # Navigate to the Sci-Hub website
        self.driver.get('https://www.sci-hub.se/')
        time.sleep(5)  # Allow the page to load completely

        # Loop through each DOI in the provided list
        for doi in tqdm(dois, desc="Downloading papers"):
            # Skip if the DOI is not found
            if doi == 'Not Found': 
                self.downloaded_files.append("No pdf")
                continue

            # Locate the search text box on the Sci-Hub page and enter the DOI
            text_box = self.driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/form/div/textarea')
            text_box.clear()  # Clear any existing text in the search box
            text_box.send_keys(doi)  # Input the DOI into the search box
            
            # Locate and click the search button
            search_button = self.driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/form/div/button')
            search_button.click()
            time.sleep(3)  # Wait for the results to load

            # Try to locate and click the download button for the PDF
            try:
                download_button = self.driver.find_element(By.XPATH, '/html/body/div[3]/div[1]/button')
                download_button.click()  # Initiate the download
                time.sleep(5)  # Wait for the download to complete properly
                
                # Identify the most recently downloaded file in the download directory
                list_of_files = glob.glob(os.path.join(self.download_directory, '*.pdf'))
                latest_file = max(list_of_files, key=os.path.getctime)  # Find the latest file by creation time
                file_name = os.path.basename(latest_file)  # Extract the file name
                self.downloaded_files.append(file_name)  # Append the file name to the list of downloaded files

            # Handle case where the download button is not found
            except NoSuchElementException:
                print("Button not found on the page.")
                self.downloaded_files.append("No pdf")
            
            # Go back to the previous page to search for the next DOI
            self.driver.back()
            time.sleep(2)  # Wait for the page to load

        # Close the WebDriver after all downloads are complete
        self.close()

        # Return the list of downloaded files
        return self.downloaded_files

    # Method to close the WebDriver
    def close(self):
        if self.driver:
            # Quit the WebDriver if it has been initialized
            self.driver.quit()
