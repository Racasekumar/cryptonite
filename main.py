import os
import re
import requests
import together
from together import Together
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()

client = Together(api_key = os.environ['TOGETHER_API_KEY'])
# together.api_key = os.getenv("together_key")

model = "meta-llama/Meta-Llama-3.1-8B-Instruct-lora"


class Agent:
    def __init__(self, client: Together, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result.strip().lower()  # Return normalized output

    def execute(self):
        completion = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-lora", messages=self.messages
        )
        # Print raw output for debugging
        print("Agent response:", completion.choices[0].message.content)
        return completion.choices[0].message.content

# Define system prompt for the agent
system_prompt = """
You can ask me a question in any language, and I'll respond with the single word cryptocurrency name in English.
Here's how it works:

1. If your question is not in English, I'll translate it into English.
2. I'll extract the single word cryptocurrency name from your question.
3. I'll respond with the single word cryptocurrency name in English.

Available cryptocurrencies: bitcoin, ethereum, litecoin, etc.

Go ahead and ask your question!
"""

# Function to fetch crypto price
def get_crypto_price(crypto_symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_symbol}&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if crypto_symbol in data:
            price = data[crypto_symbol]["usd"]
            return price
    return None  # Return None if price or crypto not found

# Initialize the agent
cryptonite = Agent(client=client, system=system_prompt)

# Ask for user input
query = input("Ask me about a cryptocurrency: ")

# Get the result from the agent
crypto_name = cryptonite(query)

# Remove any extraneous characters (e.g., quotes) around the response
crypto_name = crypto_name.strip().strip("'\"").strip()
# Try to get the last word from the response which should be the cryptocurrency name
match = re.search(r"(\w+)$", crypto_name)
if match:
    crypto_name = match.group(1)

# Check if a cryptocurrency was extracted and fetch its price
if crypto_name:
    price = get_crypto_price(crypto_name)
    if price:
        print(f"The current price of {crypto_name.capitalize()} is ${price} USD.")
    else:
        print(f"Could not retrieve the price for '{crypto_name}'. Please try with another cryptocurrency.")
else:
    print("Could not determine a cryptocurrency from the input.")
