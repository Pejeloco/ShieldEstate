# ShieldEstate: Ensuring Your Rent

ShieldEstate; an app where anyone seeking to rent can ensure a competitive price in Madrid!!! Including an nteractive map for you where you can vizualize the distribution of predicted prices for Airbnb all over Madrid!


## Description: 


ShieldEstate has created an app capable of offering your estate's predicted price per night and evaluating the price you had in mind relative to the location and other key attributes. Through our machine learning algorithm which trains the price ranges in different regions our script predicts the price of an Airbnb listing in Madrid Centro based on the number of guests, whether the host is a superhost, and the listing's location (latitude and longitude). With the previous information our app generates a heatmap around Madrid displaying the predicted prices for estates in different locations with your selected attributes. 

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

### Prerequisites

Ensure that you have the following installed on your system:

- Python 3.x
- Required Python libraries: pandas, numpy, folium, scikit-learn, jinja2, branca

### Setup

- Clone the repository or download the source code.
- Ensure that all the files, including the dataset file are in the same working directory.

### Usage

- Run the script using the following command: python stratup_main.py
- Follow the on-screen prompts to input information about your listing
- The script will predict the price for your listing, evaluate your suggested price, and generate a heatmap that includes your listing
- The heatmap will be saved as madrid_centro_heatmap.html in your current directory. Open this file in a web browser to view the heatmap.
- If you want to save multiple heatmaps, rename the existing madrid_centro_heatmap.html file before running the script again


## App Architecture

The Airbnb Madrid Price Prediction and Heatmap app consist of the following main components:

1. **User Authentication**: Securely handling user login and registration, ensuring only authorized users have access to the app's features.

2. **Data Preprocessing**: Importing and cleaning the dataset, including removing outliers and converting data types as necessary.

3. **Model Training**: Training a RandomForestRegressor model on the preprocessed data, using GridSearchCV for hyperparameter tuning.

4. **Price Prediction**: Using the trained model to predict the price of a new Airbnb listing based on the number of guests, whether the host is a superhost, and the location (latitude and longitude).

5. **Price Evaluation**: Comparing the predicted price with the user's suggested price and providing a price suggestion category (A, B, C, or D). If the user's suggested price has a percentage difference ± 5, it's ranked as category A. If the percentage difference ± 10, it's ranked as category B. If the percentage difference ± 20, it's ranked as category C. Else, it's ranked as category D.

6. **Heatmap Generation**: Creating an interactive heatmap that visualizes the distribution of predicted prices for Airbnb listings in Madrid, including the user's listing.


## Resources Used
The following resources were used in the development of this code:

- Python 3
- Pandas
- NumPy
- Folium
- Folium.plugins
- Folium.map
- Scikit-learn
- Sklearn.ensemble
- Sklearn.preprocessing
- Sklearn.model_selection
- Branca


## Credits:

This project was created for our Algorithms and Data Structures course at IE University. The project was created by:

1. **Brian Collado**: 
https://github.com/Bcollado0310 // https://www.linkedin.com/me?trk=p_mwlite_feed_updates-secondary_nav

2. **Andres Gerli**: 
https://github.com/Pejeloco // https://www.linkedin.com/in/andres-gerli-29956422b/

3. **Jaime Berasategui**: 
https://github.com/Jaimeberasategui // https://www.linkedin.com/in/jaime-berasategui-4b0a5b254/
4. **Jose Urgal**: 
https://github.com/Joseurgal // https://www.linkedin.com/in/jos%C3%A9-urgal-saracho-255b99250/

## How you can contribute to this project:

Mainly, try to make webscraping of airbnb, but from other places. We suggest:
- Barcelona
- Valencia
- Sevilla
- Bilbao

Moreover, another way to improve our project is by including a user interface.
