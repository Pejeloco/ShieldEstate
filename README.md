# ShieldEstate: Ensuring Your Rent

ShieldEstate; an app where anyone seeking to rent can ensure a competitive price in Madrid!!! Including an nteractive map for you where you can vizualize the distribution of predicted prices for Airbnb all over Madrid!


##Desccription: 

ShieldEstate has created an app capable of offering your estate's predicted price per night and evaluating the price you had in mind relative to the location and other key attributes. Through our machine learning algorithm which trains the price ranges in different regions our script predicts the price of an Airbnb listing in Madrid Centro based on the number of guests, whether the host is a superhost, and the listing's location (latitude and longitude). With the previous information our app generates a heatmap around Madrid displaying the predicted prices for estates in different locations with your selected attributes. '

##Key Features

  Trains a RandomForestRegressor model on historical Airbnb data to predict listing prices.
  
  Prompts the user for information about their listing (number of guests, superhost status, latitude, and longitude).
  
  Predicts the price for the user's listing and evaluates how it compares to the suggested price.
  Generates an interactive heatmap that visualizes the distribution of predicted prices, including the user's listing.
  
  Saves the heatmap as an HTML file.
