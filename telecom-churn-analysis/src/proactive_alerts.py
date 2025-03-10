import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(location, issue):
    sender_email = "you@example.com"
    receiver_email = "stakeholder@example.com"
    password = "your_password"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Proactive Alert: Network Issue in " + location
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"""
    Hi Team,

    Please be informed that there is a network issue in {location}. The following issue has been detected:
    {issue}

    Kindly take the necessary actions.

    Best Regards,
    Network Monitoring Team
    """
    part1 = MIMEText(text, "plain")
    message.attach(part1)

    with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

def check_network_performance(data):
    for index, row in data.iterrows():
        if row['Network Performance Score'] < 6:
            send_alert(row['Location'], "Network Performance Score below 6")
        if row['Predicted Churn Risk'] > 0.2:
            send_alert(row['Location'], "Predicted Churn Risk above 20%")