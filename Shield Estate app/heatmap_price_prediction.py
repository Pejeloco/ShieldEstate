#We import the neccesary libraries
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import branca
import pickle
import os
import webbrowser

#We create the class
class PredictionPriceHeatMap:
    def __init__(self): #we create the init method and then we pass it, this because we want to run it in the python interpreter, and we need to pass the class to do this.
        pass 

    @staticmethod #We create the static method to load the data, this method will be used in the main file. Thats is why we use the static method
    def train_model(data): #We create the method to train the model, we pass the data as a parameter
        data['isHostedBySuperhost'] = data['isHostedBySuperhost'].astype(int) #We convert the column isHostedBySuperhost to int, that way we can use it in the model, and change the True and False values to 1 and 0

        #Now we remove the outliers from the data, this will give us a better model
        Q1 = data['pricing/rate/amount'].quantile(0.25)
        Q3 = data['pricing/rate/amount'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #We remove the outliers from the pricing/rate/amount column in the dataset. We use the lower and upper bound to do this.
        data = data[(data['pricing/rate/amount'] >= lower_bound) & (data['pricing/rate/amount'] <= upper_bound)]

        #We create the features and target variables
        features = ['numberOfGuests', 'isHostedBySuperhost', 'location/lat', 'location/lng']
        target = 'pricing/rate/amount'

        #we create the X and y variables, We do it as a copy of the data, so we dont modify the original data, and we can replicate the process without having to load the data again.
        X = data[features].copy()
        y = data[target].copy()

        #We split the data into train and test, we use 50% of the data for testing, and 50% for training. This is because of the size of our dataset. Since it's only 2000 rows, we need to use as much data as possible to train the model.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # This part is important because we need to make sure that the columns in the test data are the same as the columns in the train data. If we dont do this, we will get an error when we try to scale the data.
        scaler = StandardScaler() #we create the scaler object
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features) #we fit the scaler to the training data, and then we transform it. We do the same with the test data, but we only transform it.
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features) #we fit the scaler to the test data, and then we transform it. We do the same with the test data, but we only transform it.

        #For model we deciced to use the Random Forest Regressor, because it is a good, if not the best model for this type of problem. 
        model = RandomForestRegressor(random_state=42)

        #We create the param_grid, this is the grid that we will use to find the best parameters for the model. We use the GridSearchCV to do this.
        param_grid = {
            'n_estimators': [10, 50, 100, 200], #We use different values for the n_estimators, this is the number of trees in the forest.
            'max_depth': [None, 10, 20, 30], #We use different values for the max_depth, this is the maximum depth of the tree.
            'min_samples_split': [2, 5, 10], #We use different values for the min_samples_split, this is the minimum number of samples required to split an internal node.
            'min_samples_leaf': [1, 2, 4], #We use different values for the min_samples_leaf, this is the minimum number of samples required to be at a leaf node.
            'max_features': ['auto', 'sqrt'] #We use different values for the max_features, this is the number of features to consider when looking for the best split.
        }

        #We create the grid_search object, and we fit it to the training data. This will give us the best model.
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        #We get the best model from the grid_search object, and we fit it to the training data.
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        #We return the best model and the scaler, so we can use them later.
        return best_model, scaler
    
    #This function is to save the model and the scaler to a pickle file, so we can use them later. This way, the user doesnt have to train the model every time they want to use it.
    def save_model_to_pickle(model, scaler, filename='best_model.pkl'):
        model_scaler_data = {
            'model': model,
            'scaler': scaler
        }

        with open(filename, 'wb') as f: #We open the file in write mode, and we save the model and the scaler to the file.
            pickle.dump(model_scaler_data, f) #We use the pickle.dump function to save the model and the scaler to the file.

    #this function is to load the model and the scaler from the pickle file, so we can use them later. This way, the user doesnt have to train the model every time they want to use it.
    @staticmethod
    def load_model_from_pickle(filename='best_model.pkl'):
        with open(filename, 'rb') as f:
            model_scaler_data = pickle.load(f)

        model = model_scaler_data['model'] #We get the model from the model_scaler_data dictionary.
        scaler = model_scaler_data['scaler'] #We get the scaler from the model_scaler_data dictionary.

        return model, scaler #We return the model and the scaler.

    #This function is to evaluate the price difference between the predicted price and the suggested price.
    @staticmethod
    def evaluate_price_difference(predicted_price, suggested_price): #We get the predicted price and the suggested price as parameters.
        difference = abs(predicted_price - suggested_price) #e calculate the difference between the predicted price and the suggested price. the abs is to make sure that the difference is positive.
        percentage_difference = (difference / predicted_price) * 100 #We calculate the percentage difference between the predicted price and the suggested price.

        if percentage_difference <= 5:
            return "A" #If the percentage difference is less than or equal to 5, we return A.
        elif percentage_difference <= 10:
            return "B" #If the percentage difference is less than or equal to 10, we return B.
        elif percentage_difference <= 20:
            return "C" #If the percentage difference is less than or equal to 20, we return C.
        else:
            return "D" #If the percentage difference is greater than 20, we return D.
        
    #this function is to genereate the heatmap, and add the markers to the map.
    @staticmethod
    def create_heatmap(data, predictions): #We get the data and the predictions as parameters.
        map_center = [40.4168, -3.7038]  # Latitude and longitude of Madrid Centro
        m = folium.Map(location=map_center, zoom_start=14) #We create the map, and we set the location and the zoom level.

        locations = data[['location/lat', 'location/lng']].values.tolist() #We get the locations from the data, and we convert it to a list. We do this because the HeatMap function needs a list of locations.

        heatmap = HeatMap(locations, radius=15, blur=20) #We create the heatmap, and we set the radius and the blur.
        color_range = [0.2, 0.4, 0.6, 1] #We set the color range. This is the range of the colors that will be used for the markers. The colors will be from blue to red.
        color_scheme = ['blue', 'purple', 'orange', 'red'] #We set the color scheme. This is the colors that will be used for the markers. The colors will be from blue to red.

        colormap = branca.colormap.StepColormap( #We create the colormap, and we set the colors and the index.
            colors=color_scheme,
            index=color_range,
            vmin=min(color_range),
            vmax=max(color_range),
            caption='Number of Airbnb listings'
        )

        # Add predicted price markers to the map
        for index, row in predictions.iterrows(): #We iterate through the predictions dataframe.
            folium.CircleMarker( #We create the markers for the predictions.
                location=[row['location/lat'], row['location/lng']], #We set the location of the markers.
                radius=5, #We set the radius of the markers.
                popup=f"Predicted price: €{row['predicted_price']:.2f}", #We set the popup of the markers.
                fill=True, #We set the fill of the markers.
                color=colormap(row['predicted_price'] / predictions['predicted_price'].max()), #We set the color of the markers.
                fill_opacity=0.7
            ).add_to(m)

        heatmap.add_to(m)
        m.add_child(colormap)
        m.add_child(folium.map.LayerControl())

        folium.LatLngPopup().add_to(m)

        folium.TileLayer('openstreetmap').add_to(m)
        folium.LayerControl().add_to(m)

        return m
    
    

    @staticmethod
    def get_integer_input(prompt): #We create the function to get the integer input from the user, we get the prompt as a parameter.
        while True:
            try: #We use the try except block to make sure that the user enters an integer.
                value = int(input(prompt))
                if value > 0:
                    break
                else:
                    print("Invalid input. Please enter a positive integer.")
            except ValueError: #If the user enters something that is not an integer, we print an error message.
                print("Invalid input. Please enter an integer.")
        return value
    
    @staticmethod
    def get_yes_no_input(prompt): #We create the function to get the yes or no input from the user, we get the prompt as a parameter.
        while True:
            value = input(prompt).lower()
            if value in ['yes', 'no']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.") #If the user enters something that is not yes or no, we print an error message.
        return value
    
    @staticmethod
    def get_float_input(prompt): #We create the function to get the float input from the user, we get the prompt as a parameter.
        while True:
            try:
                value = float(input(prompt)) #We use the try except block to make sure that the user enters a float.
                break
            except ValueError: #If the user enters something that is not a float, we print an error message.
                print("Invalid input. Please enter a floating-point number.")
        return value


    @staticmethod
    def main(): #We create the main function, this is the function that will be called when we run the file.
        data = pd.read_excel('dataset Airbnb full.xlsx') #We load the data from the excel file.

        model_filename = 'best_model.pkl' #We set the model filename.
        if os.path.exists(model_filename): #We check if the model file exists.
            model, scaler = PredictionPriceHeatMap.load_model_from_pickle(model_filename) #If the model file exists, we load the model and the scaler from the pickle file.
        else: #If the model file doesnt exist, we train the model and the scaler.
            model, scaler = PredictionPriceHeatMap.train_model(data)
            PredictionPriceHeatMap.save_model_to_pickle(model, scaler)

        # Get predicted prices for all listings
        X = data[['numberOfGuests', 'isHostedBySuperhost', 'location/lat', 'location/lng']] #We get the features from the data.
        X_scaled = scaler.transform(X) #We scale the features because we need to scale the features before we can use them in the model.
        data['predicted_price'] = model.predict(X_scaled) #We predict the prices for the listings.

        while True: #We use a while loop to keep asking the user for input until they enter 'no'. Because we want to give the user the option to evaluate multiple listings.
            number_of_guests = PredictionPriceHeatMap.get_integer_input("Enter the number of guests: \n")
            superhost_input = PredictionPriceHeatMap.get_yes_no_input("Is the host a superhost? (yes/no): \n")
            is_hosted_by_superhost = 1 if superhost_input == "yes" else 0
            location_latitude = PredictionPriceHeatMap.get_float_input("Enter the location's latitude: \n")
            location_longitude = PredictionPriceHeatMap.get_float_input("Enter the location's longitude: \n")

            # Predict price for the new listing
            new_listing = np.array([[number_of_guests, is_hosted_by_superhost, location_latitude, location_longitude]]) #We create the new listing array. We use the values that the user entered.
            new_listing_scaled = scaler.transform(new_listing) #We scale the new listing because we need to scale the new listing before we can use it in the model.
            predicted_price = model.predict(new_listing_scaled)[0] #We predict the price for the new listing.

            # Evaluate price difference between predicted price and suggested price
            print(f"Predicted price for the new listing per night: €{predicted_price:.2f}\n") #We print the predicted price for the new listing.
            suggested_price = PredictionPriceHeatMap.get_float_input("Enter your expected price per night: \n") #We get the suggested price from the user.
            evaluation = PredictionPriceHeatMap.evaluate_price_difference(predicted_price, suggested_price) #We evaluate the price difference between the predicted price and the suggested price.
            print(f"Your price suggestion is in category {evaluation}. \n") #We print the evaluation. If the evaluation is A, B, C or D, we print the evaluation. If the evaluation is not A, B, C or D, we print an error message.


            # Add user's listing to the dataset
            user_listing = pd.DataFrame({'numberOfGuests': [number_of_guests], #We create the user listing dataframe. We use the values that the user entered.
                                        'isHostedBySuperhost': [is_hosted_by_superhost], #We create the user listing dataframe. We use the values that the user entered.
                                        'location/lat': [location_latitude],
                                        'location/lng': [location_longitude],
                                        'predicted_price': [predicted_price]}) 
            combined_data = pd.concat([data, user_listing], ignore_index=True) #We add the user listing to the data.

            # Create and save the heatmap with user's listing
            create_heatmap_input = PredictionPriceHeatMap.get_yes_no_input("Do you want to create a heatmap with your listing? (yes/no): ").lower()  # We ask the user if they want to create a heatmap with their listing and convert their input to lowercase.
            if create_heatmap_input == "yes":  # If the user enters yes, we create the heatmap.
                heatmap = PredictionPriceHeatMap.create_heatmap(data, user_listing)
                heatmap.save('madrid_centro_heatmap.html')  # We save the heatmap as a html file.
                print("Heatmap with price suggestions and your listing saved as 'madrid_centro_heatmap.html' in your current directory. If you want to save multiple heatmaps, please rename the file before generating a new one. \n")
                print("The heatmap will appear automatically in your screen. \n")
                print("Your Location would be marked with a Red circle. \n")
                # Open the saved heatmap in the user's default web browser
                webbrowser.open('file://' + os.path.realpath('madrid_centro_heatmap.html'))
            if create_heatmap_input == "no":  # If the user enters no, we print a message.
                print("Heatmap not created. \n")

            # Ask the user if they want to evaluate another listing
            another_listing = PredictionPriceHeatMap.get_yes_no_input("Do you want to evaluate another listing? (yes/no): ")
            if another_listing == "no":
                break #If the user enters no, we break out of the while loop.

if __name__ == "__main__": #We use the if __name__ == "__main__": to make sure that the code inside the if statement is only executed when we run the file. If we import the file, the code inside the if statement will not be executed.
    PredictionPriceHeatMap.main() #We call the main function to run the code.