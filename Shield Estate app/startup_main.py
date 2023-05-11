#we import the necessary modules from the other files in order to run the program.
from signup_login import SignUp
from heatmap_price_prediction import PredictionPriceHeatMap

#we create a main function to run the program.
def main():
    while True: #we create a while loop to allow the user to choose between the options.
        print("Welcome to the Shield Estate App!\n")
        print("1. Sign Up\n")
        print("2. Login\n")
        print("3. Exit\n")
        try: #we use try and except to catch any errors.
            choice = int(input("Enter your choice (1/2/3): ")) #we ask the user to input their choice.
            if choice == 1: #if the user chooses 1, they will be asked to sign up.
                SignUp.sign_up()
            elif choice == 2: #if the user chooses 2, they will be asked to log in.
                SignUp.log_in()
                if SignUp.logged_in: #if the user has logged in successfully, they will be able to choose between the options.
                    while True: #we create a while loop to allow the user to choose between the options.
                        print("Welcome to the Shield Estate App!\n")
                        print("Choose an option:\n")
                        print("1. Price Prediction Model") #we print the options for the user to choose from.)
                        print("2. Exit\n")
                        try: #we use try and except to catch any errors.
                            choice = int(input("Enter your choice (1/2): \n"))
                            if choice == 1: #if the user chooses 1, they will be able to see the prediction price with heatmap.
                                print("This choice takes a while to load, hold tight...\n")
                                PredictionPriceHeatMap.main()
                            elif choice == 2: #if the user chooses 3, they will be able to exit the program.
                                print("Exiting...\n")
                                break
                            else: #if the user chooses an invalid choice, they will be asked to try again.
                                print("Invalid choice. Try again.\n")
                        except ValueError: #if the user inputs an invalid input, they will be asked to try again.
                            print("Invalid input. Try again.\n")
            elif choice == 3: #if the user chooses 3, they will be able to exit the program.
                break
            else:
                print("Invalid choice. Try again.\n")
        except ValueError: #if the user inputs an invalid input, they will be asked to try again.
            print("Invalid input. Try again.\n")

    print("Thank you for using the Shield Estate App.\n")

if __name__ == '__main__': #we call the main function to run the program.
    main()
