users = {}

class SignUp:
    def __init__(self):
        pass

    logged_in = False

    @staticmethod
    def log_in():
        username = input('Log In User Name: ')
        password = input('Log In Password: ')

        if username in users and users[username]['password'] == password:
            print('Logged in successfully.')
            SignUp.logged_in = True
        else:
            print('Invalid username or password.')

    @staticmethod
    def sign_up():
        username = input('Sign Up Username: ')

        # Check if the username is already taken
        if username in users:
            print('This username is already taken.')
            return

        password = input('Sign Up Password: ')
        users[username] = {'username': username, 'password': password}
        print('Account created successfully.')

    @staticmethod
    def main():
        while True:
            print('1. Log In')
            print('2. Sign Up')
            print('3. Exit')
            choice = input('Enter your choice (1/2/3): ')

            if choice == '1':
                SignUp.log_in()
                if SignUp.logged_in:
                    break
            elif choice == '2':
                SignUp.sign_up()
            elif choice == '3':
                print('Exiting...')
                break
            else:
                print('Invalid choice. Please try again.')

if __name__ == '__main__':
    SignUp.main()
