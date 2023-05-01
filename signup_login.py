import logging
import random
import string
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

users = {}

class SignUp:
    def __init__(self):
        pass

    @staticmethod
    def send_verification_email(email, verification_code):
        api_key = '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'

        message = Mail(
            from_email='groupshieldestate@gmail.com',
            to_emails=email,
            subject='Shield Estate Account Verification',
            html_content=f'<strong>Hello,</strong><br><br>Your verification code for Shield Estate is: {verification_code}<br><br>Thank you for choosing Shield Estate!'
        )

        try:
            sg = SendGridAPIClient(api_key)
            response = sg.send(message)
        except Exception as e:
            print(f'Error sending email: {e}')

    @staticmethod
    def generate_verification_code():
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    logged_in = False


    @staticmethod
    def log_in():
        email = input('Log In Email: ')
        password = input('Log In Password: ')

        if users.get(email) and users[email]['password'] == password:
            print('Logged in successfully.')
            SignUp.logged_in = True  # Change this line
        else:
            print('Invalid email or password.')


    @staticmethod
    def sign_up():
        username = input('Sign Up Username: ')
        email = input('Sign Up Email: ')

        # Check if the username is already taken
        for user_data in users.values():
            if user_data['username'] == username:
                print('This username is already taken.')
                return

        password = input('Sign Up Password: ')
        verification_code = SignUp.generate_verification_code()
        SignUp.send_verification_email(email, verification_code)

        print('A verification code has been sent to your email.')

        user_verification_code = input('Enter the verification code: ')
        if user_verification_code == verification_code:
            users[email] = {'username': username, 'password': password}
            print('Account created successfully.')
        else:
            print('Incorrect verification code. Account not created.')


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