from email.message import EmailMessage
import ssl
import smtplib

email_sender = "alexli9133@gmail.com"
email_password = "" # check discord for email pw, put in .env when integrated
email_receiver = "jajooananya@gmail.com" # works w/ any test email

subject = 'code'
body = """
hello ananya this is a test. you have reached a safety breach.
"""

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['Subject'] = subject
em.set_content(body)

context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
    smtp.login(email_sender, email_password)
    smtp.sendmail(email_sender, email_receiver, em.as_string())
