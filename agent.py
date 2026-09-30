import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#MODEL = "qwen/qwen3.8-27b"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def run():
    '''Run the agent's conversation loop until the users quit'''

    # The conversation history. This is the entire memory of the agent
    # Every turn, we append to it and send the whole thing to the model
    messages =[]

    print("Agent ready. Type 'quit' or 'exit' to leave\n")

    while True:
        # 1. Get input from the user.
        user_input = input("you > ").strip()

        #2. Allow the user to leave cleanly
        if user_input in {"quit","exit"}:
            print("So long")
            break

        # Skip lines without making a model call
        if not user_input:
            continue

        #3. Append the user's message to the history
        messages.append({
            "role": "user",
            "content": user_input
        })

        #4. Call the model with the full conversation so far.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        # 5. Extract the assistant's reply
        assistant_message = response.choices[0].message.content

        #6. Append the assistants's reply to the history
        messages.append({"role": "assistant","content": assistant_message})

        #7. Show thr user
        print(f"\nagent > {assistant_message}")

if __name__ == "__main__":
    run()
