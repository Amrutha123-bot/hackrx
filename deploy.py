from dotenv import load_dotenv
import os

load_dotenv()
heroku_key = os.getenv('HEROKU_API_KEY')

# You can now use heroku_key variable wherever needed
print(f"My Heroku API Key is {heroku_key}")  # Just for testing, remove in production!
