import asyncio
import os
import ssl
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import sys

load_dotenv()

email_sender = "alexli9133@gmail.com"
email_password = os.getenv("EMAIL_PASSWORD")
email_receiver = "pranav.neti108@gmail.com"

async def send_alert_email(video_name: str):
    subject = "ALERT!! ALERT!!"
    body = f"""
    THERE HAS BEEN A SAFETY BREACH

    Video - {video_name}
    """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    def send():
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(em)

    await asyncio.to_thread(send)

if __name__ == '__main__':
    video_name = sys.argv[1]  # Get video filename from command-line argument
    asyncio.run(send_alert_email(video_name))