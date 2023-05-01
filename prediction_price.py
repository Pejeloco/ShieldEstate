import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class PredictionPrice:
    def __init__(self):
        pass

    @staticmethod
    def train_model():
        data = pd.read_excel('dataset Airbnb full.xlsx')
        data['isHostedBySuperhost'] = data['isHostedBySuperhost'].astype(int)

        Q1 = data['pricing/rate/amount'].quantile(0.25)
        Q3 = data['pricing/rate/amount'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data = data[(data['pricing/rate/amount'] >= lower_bound) & (data['pricing/rate/amount'] <= upper_bound)]

        features = ['numberOfGuests', 'isHostedBySuperhost', 'location/lat', 'location/lng']
        target = 'pricing/rate/amount'

        X = data[features].copy()
        y = data[target].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        return best_model, scaler

    @staticmethod
    def evaluate_price_difference(predicted_price, suggested_price):
        difference = abs(predicted_price - suggested_price)
        percentage_difference = (difference / predicted_price) * 100

        if percentage_difference <= 5:
            return "A"
        elif percentage_difference <= 10:
            return "B"
        elif percentage_difference <= 20:
            return "C"
        else:
            return "D"

    @staticmethod
    def main():
        model, scaler = PredictionPrice.train_model()

        while True:
            number_of_guests = int(input("Enter the number of guests: \n"))
            superhost_input = input("Is the host a superhost? (yes/no): \n").lower()
            is_hosted_by_superhost = 1 if superhost_input == "yes" else 0
            location_latitude = float(input("Enter the location's latitude: \n"))
            location_longitude = float(input("Enter the location's longitude: \n"))

            new_listing = np.array([[number_of_guests, is_hosted_by_superhost, location_latitude, location_longitude]])
            new_listing_scaled = scaler.transform(new_listing)
            predicted_price = model.predict(new_listing_scaled)[0]

            print(f"Predicted price for the new listing: €{predicted_price:.2f}\n")
            suggested_price = float(input("Enter your suggested price: \n"))
            evaluation = PredictionPrice.evaluate_price_difference(predicted_price, suggested_price)
            print(f"Your price suggestion is in category {evaluation}.\n")

            another_listing = input("Do you want to evaluate another listing? (yes/no): \n").lower()
            if another_listing == "no":
                break

if __name__ == "__main__":
    PredictionPrice.main()