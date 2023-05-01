import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from folium.map import CustomPane
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import branca


class CustomControl(folium.map.CustomPane):
    def __init__(self, name, html, *args, **kwargs):
        super(CustomControl, self).__init__(name, *args, **kwargs)
        self.html = html

    def render(self, **kwargs):
        super(CustomControl, self).render(**kwargs)
        self.content = self.html.format(self=self, kwargs=kwargs)


class PredictionPriceHeatMap:
    def __init__(self):
        pass

    @staticmethod
    def train_model(data):
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
    def create_heatmap(data, predictions):
        map_center = [40.4168, -3.7038]  # Latitude and longitude of Madrid Centro
        m = folium.Map(location=map_center, zoom_start=14)

        locations = data[['location/lat', 'location/lng']].values.tolist()

        heatmap = HeatMap(locations, radius=15, blur=20)
        color_range = [0.2, 0.4, 0.6, 1]
        color_scheme = ['blue', 'purple', 'orange', 'red']

        colormap = branca.colormap.StepColormap(
            colors=color_scheme,
            index=color_range,
            vmin=min(color_range),
            vmax=max(color_range),
            caption='Number of Airbnb listings'
        )

        # Add predicted price markers to the map
        for index, row in predictions.iterrows():
            folium.CircleMarker(
                location=[row['location/lat'], row['location/lng']],
                radius=5,
                popup=f"Predicted price: €{row['predicted_price']:.2f}",
                fill=True,
                color=colormap(row['predicted_price'] / predictions['predicted_price'].max()),
                fill_opacity=0.7
            ).add_to(m)

        heatmap.add_to(m)
        m.add_child(colormap)
        m.add_child(folium.map.LayerControl())

        folium.LatLngPopup().add_to(m)

        folium.TileLayer('openstreetmap').add_to(m)
        folium.LayerControl().add_to(m)

        map_id = m.get_name()

        global_map_script = f"<script>var globalMap = maps['{map_id}'];</script>"
        m.get_root().header.add_child(folium.Element(global_map_script))

        custom_button_html = """
            <div id="{{kwargs['map_id']}}_{{self.get_name()}}_control" class="custom-center-control" onclick="goToUserLocation()">Go to your listing</div>
            <script>
                function goToUserLocation() {{
                    if (typeof {{kwargs['map_id']}} !== 'undefined') {{
                        var userLat = "{{predictions.iloc[-1]['location/lat']}}";
                        var userLng = "{{predictions.iloc[-1]['location/lng']}}";
                        {{kwargs['map_id']}}.setView([userLat, userLng], 18);
                    }}
                }}
            </script>
            """


        custom_button_control = CustomControl(name="CustomButton", html=custom_button_html)
        m.add_child(custom_button_control)

        return m
    
    @staticmethod
    def get_integer_input(prompt):
        while True:
            try:
                value = int(input(prompt))
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")
        return value
    
    def get_yes_no_input(prompt):
        while True:
            value = input(prompt).lower()
            if value in ['yes', 'no']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        return value


    @staticmethod
    def main():
        data = pd.read_excel('dataset Airbnb full.xlsx')
        model, scaler = PredictionPriceHeatMap.train_model(data)

        # Get predicted prices for all listings
        X = data[['numberOfGuests', 'isHostedBySuperhost', 'location/lat', 'location/lng']]
        X_scaled = scaler.transform(X)
        data['predicted_price'] = model.predict(X_scaled)

        while True:
            number_of_guests = PredictionPriceHeatMap.get_integer_input("Enter the number of guests: \n")
            superhost_input = PredictionPriceHeatMap.get_yes_no_input("Is the host a superhost? (yes/no): \n")
            is_hosted_by_superhost = 1 if superhost_input == "yes" else 0
            location_latitude = float(input("Enter the location's latitude: \n"))
            location_longitude = float(input("Enter the location's longitude: \n"))

            new_listing = np.array([[number_of_guests, is_hosted_by_superhost, location_latitude, location_longitude]])
            new_listing_scaled = scaler.transform(new_listing)
            predicted_price = model.predict(new_listing_scaled)[0]

            print(f"Predicted price for the new listing per night: €{predicted_price:.2f}\n")
            suggested_price = float(input("Enter your expected price per night: \n"))
            evaluation = PredictionPriceHeatMap.evaluate_price_difference(predicted_price, suggested_price)
            print(f"Your price suggestion is in category {evaluation}. \n")

            # Add user's listing to the dataset
            user_listing = pd.DataFrame({'numberOfGuests': [number_of_guests],
                                        'isHostedBySuperhost': [is_hosted_by_superhost],
                                        'location/lat': [location_latitude],
                                        'location/lng': [location_longitude],
                                        'predicted_price': [predicted_price]})
            combined_data = pd.concat([data, user_listing], ignore_index=True)

            # Create and save the heatmap with user's listing
            heatmap = PredictionPriceHeatMap.create_heatmap(data, user_listing)
            heatmap.save('madrid_centro_heatmap.html')
            print("Heatmap with price suggestions and your listing saved as 'madrid_centro_heatmap.html' in your current directory. If you want to save multiple heatmaps, please rename the file before generating a new one. \n")
            print("You can open the heatmap in your browser by double-clicking the file. \n")
            print("Your Location would be marked with a Red circle. \n")

            another_listing = input("Do you want to evaluate another listing? (yes/no): ").lower()
            if another_listing == "no":
                break


if __name__ == "__main__":
    PredictionPriceHeatMap.main()