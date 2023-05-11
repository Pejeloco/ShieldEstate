
![ShieldEstate](https://user-images.githubusercontent.com/126095106/235503538-04492a65-e4c1-42e0-85b0-9d5529f416d0.jpg)

# ShieldEstate: Ensuring Your Rent

ShieldEstate; an app where anyone seeking to rent can ensure a competitive price in Madrid!!! Including an nteractive map for you where you can vizualize the distribution of predicted prices for Airbnb all over Madrid!


## Description: 


ShieldEstate has created an app capable of offering your estate's predicted price per night and evaluating the price you had in mind relative to the location and other key attributes. Through our machine learning algorithm which trains the price ranges in different regions our script predicts the price of an Airbnb listing in Madrid Centro based on the number of guests, whether the host is a superhost, and the listing's location (latitude and longitude). With the previous information our app generates a heatmap around Madrid displaying the predicted prices for estates in different locations with your selected attributes. 

## Table of Contents

- Key Features
- Limitations
- [Installation & Usage](#installation--usage)
- - [Prerequisites](#prerequisites)
- - [Setup](#setup)
- - [Usage](#usage)
- [App Architecture](#app-architecture)
- [Resources Used](#resources-used)
- [Credits](#credits)
- [How to Contribute to this project](#how-to-contribute-to-this-project)

## Key Features

<span style="font-size: 14px;">

- Trains a RandomForestRegressor model on historical Airbnb data to predict listing prices.
- Prompts the user for information about their listing (number of guests, superhost status, latitude, and longitude).
- Predicts the price for the user's listing and evaluates how it compares to the suggested price.
- Generates an interactive heatmap that visualizes the distribution of predicted prices, including the user's listing.
- Saves the heatmap as an HTML file.

</span>

## Limitations

The Airbnb Madrid Price Prediction and Heatmap app has some limitations that should be considered when interpreting the results:

1. **Limited dataset**: The dataset used for training the model consists of only 2,000 listings, which may not fully capture the variations and trends in Airbnb listings in Madrid.

2. **Lack of amenities data**: The dataset does not include information about amenities, which could have a significant impact on the price of an Airbnb listing.

3. **No booking schedule data**: The dataset does not provide information on the booking schedules of the listings over the years. This information would be useful to assess the acceptance rate and to better understand the risk associated with different properties.

These limitations should be taken into account when using the app to make decisions about pricing and evaluating Airbnb listings in Madrid. Future iterations of the app may include more comprehensive datasets and additional features to provide more accurate predictions and insights.

## Installation & Usage

Follow these instructions to set up and use the Airbnb Madrid Price Prediction and Heatmap app:

### Information

To execute the code it will be necessary to have installed in the system Python. The study has been done on the Windows 11 OS and the code has been written in Python 3.10.11 version.

### Prerequisites

Ensure that you have the following installed on your system:

- Python 3.x
- pip command mus be installed in order to import the following python libraries
- Required Python libraries: pandas, numpy, folium, scikit-learn, branca, pickle, os, webbrowser

### Setup

- Clone the repository or download the resources inside the Shield Estate app folder.
- Ensure that all the files, including the dataset file are in the same folder.
- Set that folder as your working directory in your terminal.

### Usage

- Open your command prompt terminal.
- Set up you terminal directory to be the folder where you store the downloaded documents.
- create a virtual environment inside with this new file directory
- Run the following line of code to install all the necessary libraries: ```pip install -r requirements.txt```
- Run the following line of code to initizalize the Shield estate app: ```python startup_main.py```
- Follow the on-screen prompts to input information about your listing
- The script will predict the price for your listing, evaluate your suggested price, and generate a heatmap that includes your listing
- The heatmap will be saved as madrid_centro_heatmap.html in your current directory. Open this file in a web browser to view the heatmap.
- If you want to save multiple heatmaps, rename the existing madrid_centro_heatmap.html file before running the script again


## App Architecture

The Airbnb Madrid Price Prediction and Heatmap app consist of the following main components:

1. **User Authentication**: Storing the Sign up user's information in a dictionary, to provide a secure login.

2. **Data Preprocessing**: Importing and cleaning the dataset, including removing outliers and converting data types as necessary.

3. **Model Training**: Training a RandomForestRegressor model on the preprocessed data, using GridSearchCV for hyperparameter tuning.

4. **Price Prediction**: Using the trained model to predict the price of a new Airbnb listing based on the number of guests, whether the host is a superhost, and the location (latitude and longitude).

5. **Price Evaluation**: Comparing the predicted price with the user's suggested price and providing a price suggestion category (A, B, C, or D). If the user's suggested price has a percentage difference ± 5, it's ranked as category A. If the percentage difference ± 10, it's ranked as category B. If the percentage difference ± 20, it's ranked as category C. Else, it's ranked as category D.

6. **Heatmap Generation**: Creating an interactive heatmap that visualizes the distribution of predicted prices for Airbnb listings in Madrid, including the user's listing.


## Resources Used
The following resources were used in the development of this code:

- Python 3
- [Pandas]([url](https://pandas.pydata.org/))
- [NumPy]([url](https://numpy.org/))
- [Folium]([url](https://pypi.org/project/folium/))
- Folium.plugins
- Folium.map
- [Scikit-learn]([url](https://scikit-learn.org/stable/))
- Sklearn.ensemble
- Sklearn.preprocessing
- Sklearn.model_selection
- [Branca]([url](https://pypi.org/project/branca/))


## Credits:

This project was created for our Algorithms and Data Structures course at IE University. The project was created by:

1. **Brian Collado**: 
https://github.com/Bcollado0310 // https://www.linkedin.com/in/brian-collado-37a333243/

2. **Andres Gerli**: 
https://github.com/Pejeloco // https://www.linkedin.com/in/andres-gerli-29956422b/

3. **Jaime Berasategui**: 
https://github.com/Jaimeberasategui // https://www.linkedin.com/in/jaime-berasategui-4b0a5b254/
4. **Jose Urgal**: 
https://github.com/Joseurgal // https://www.linkedin.com/in/jos%C3%A9-urgal-saracho-255b99250/https://www.linkedin.com/in/jos%C3%A9-urgal-saracho-255b99250/ 

## How to contribute to this project:

Mainly, try to make webscraping of airbnb, but from other places. We suggest:
- Barcelona
- Valencia
- Sevilla
- Bilbao

Moreover, another way to improve our project is by including a user interface.
