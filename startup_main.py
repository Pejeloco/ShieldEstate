from signup_login import SignUp
from heatmap_price_prediction import PredictionPriceHeatMap
from prediction_price import PredictionPrice


def main():
    while True:
        print("Welcome to the Shield Estate App\n")
        print("1. Sign Up\n")
        print("2. Login\n")
        print("3. Exit\n")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                SignUp.sign_up()
            elif choice == 2:
                SignUp.log_in()
                if SignUp.logged_in:
                    while True:
                        print("1. Prediction Price with heatmap (This choice takes a while to load)\n")
                        print("2. Prediction Price without heatmap (This choice takes a while to load)\n")
                        print("3. Exit\n")
                        try:
                            choice = int(input("Enter your choice: "))
                            if choice == 1:
                                print("This choice takes a while to load, hold tight...\n")
                                PredictionPriceHeatMap.main()
                            elif choice == 2:
                                PredictionPrice.main()
                                print("This choice takes a while to load, hold tight...\n")
                            elif choice == 3:
                                print("Exiting...\n")
                                break
                            else:
                                print("Invalid choice. Try again.\n")
                        except ValueError:
                            print("Invalid input. Try again.\n")
            elif choice == 3:
                break
            else:
                print("Invalid choice. Try again.\n")
        except ValueError:
            print("Invalid input. Try again.\n")

    print("Thank you for using the Shield Estate App.\n")

if __name__ == '__main__':
    main()
