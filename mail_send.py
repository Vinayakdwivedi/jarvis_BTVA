import smtplib
import os

# Get credentials from environment variables (set these in your OS)
# EMAIL = input("SENDER_EMAIL")


# receiver_email = input("RECEIVER EMAIL: ")
# subject = input("SUBJECT: ")
# message = input("MESSAGE: ")

# # Create email format

# print(f"📧 Sending email to {receiver_email}...")
# print(f"Subject: {subject}")
# print(f"Message: {message}")


def send_email(sender_email, receiver_email, subject, message):
    option = input("Do you want to send this email? (yes/no): ").strip().lower()
    PASSWORD = "qwkd wzsf rjeu ylde"
    if option != 'yes':
        print("❌ Email sending cancelled.")
        exit()

    else:
        try:
        # Connect to Gmail SMTP server
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, PASSWORD)
                text = f"Subject: {subject}\n\n{message}"
                server.sendmail(sender_email, receiver_email, text)
                print(f"✅ Email has been sent to {receiver_email}")

        except smtplib.SMTPAuthenticationError:
            print("❌ Authentication failed. Check your email or app password.")
        except Exception as e:
            print(f"⚠ An error occurred: {e}")
            print("✅ Email sending confirmed.")
# Ensure the script is run with Python 3
if __name__ == "__main__":
    send_email() 
