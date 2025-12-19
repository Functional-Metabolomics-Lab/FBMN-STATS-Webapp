import streamlit as st
from openai import OpenAI
from google import genai
from google.genai import types
import os
import time
import json

# Cache with st
@st.cache_data(max_entries=50)
def get_page_context(page):

    if not page:
        return None
    
    # Read the text file at assets/prompts/page.txt
    page_context_file = f"./assets/prompts/{page}.txt"
    if os.path.exists(page_context_file):
        print(f"Loading page context from {page_context_file}")
        with open(page_context_file, "r") as file:
            page_context = file.read()
        return page_context
    else:
        print(f"No page context file found for page: {page_context_file}")
        return None

def gemini_chat():
    # Setup API Key
    GEMINI_KEY = os.getenv("GOOGLEGEMINIAPI")
    if not GEMINI_KEY:
        st.error("Gemini API key not found. Please set the GOOGLEGEMINIAPI environment variable.")
        return

    # Reading preprompt
    try:
        with open("./assets/prompts/preprompt.txt", "r") as f:
            preprompt = f.read()
    except FileNotFoundError:
        preprompt = "You are a helpful assistant."

    # Initialize chat history (Consistent with OpenAI format)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    first_message = "Hello, I am here to help you write your first statistical analysis!"
    

    # Accept user input
    if prompt := st.chat_input(first_message):
        # Display user message
        # with st.chat_message("user"):
        #     st.markdown(prompt)

        # Handle page context
        page_context = get_page_context(st.session_state.get("page"))

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )


        config = types.GenerateContentConfig(
            system_instruction=preprompt,
            tools=[grounding_tool]
        )
        
        # Start client
        client = st.session_state.get("gemini_client")
        if not client:
            print("Creating a new Gemini client...", flush=True)
            client = genai.Client(api_key=GEMINI_KEY)
            st.session_state["gemini_client"] = client

        # Start Gemini Chat Session
        chat = st.session_state.get("gemini_chat_instance")
        if not chat:
            print("Creating a new chat...", flush=True)
            chat = client.chats.create(
                model='gemini-2.5-flash',
                config=config
                )
            st.session_state["gemini_chat_instance"] = chat

        try:
            # Construct the final prompt with context if available
            full_prompt = f"{page_context}\n\nUser Question: {prompt}" if page_context else prompt
            
            # Send message
            response = chat.send_message(full_prompt)
            assistant_response = response.text

            # Update Session State
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")

    for message in st.session_state.messages[::-1]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def openai4ochat():
    # readng preprompt from disk
    preprompt = open("./assets/prompts/preprompt.txt", "r").read()

    OPENAI_KEY = os.getenv("OPENAPIKEY")
    if not OPENAI_KEY:
        st.error("OpenAI API key not found. Please set the OPENAPIKEY environment variable.")
        return

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=OPENAI_KEY)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    first_message = "Hello, I am here to help you write your first statistical analysis!"
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(first_message):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        page_context = get_page_context(st.session_state.get("page"))

        if page_context:
            prompt_with_context = st.session_state.messages + \
                [{"role": "system", "content": page_context} ]+ \
                    [{"role": "user", "content": prompt}]
        else:
            prompt_with_context = st.session_state.messages + \
                [{"role": "user", "content": prompt}]

        # send user's message to GPT-4o and get a response
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": preprompt},
                *prompt_with_context
            ]
        )

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        assistant_response = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        # display GPT-4o's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Saving the chat history
        # chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
        # with open(chat_history_filename, "w") as file:
        #     file.write(json.dumps(st.session_state.messages))

def openaiassistantchat():
    OPENAI_KEY = os.getenv("OPENAPIKEY")
    if not OPENAI_KEY:
        st.error("OpenAI API key not found. Please set the OPENAPIKEY environment variable.")
        return

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=OPENAI_KEY)

    # Set a default model
    first_message = "Hello, I am here to help you write your first statistical analysis!"
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = client.beta.assistants.retrieve("asst_j5kysyB4PCwJ3sx493D0re4n")

    if "thread" not in st.session_state:
        my_thread = client.beta.threads.create()
        st.session_state["thread"] = my_thread

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input(first_message):
        if len(prompt) > 500:
            st.error("Please limit your message to 500 characters.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            my_thread_message = client.beta.threads.messages.create(
                thread_id=st.session_state["thread"].id,
                role="user",
                content=prompt,
            )

            my_run = client.beta.threads.runs.create(
                thread_id=st.session_state["thread"].id,
                assistant_id=st.session_state["assistant"].id
            )

            while my_run.status in ["queued", "in_progress"]:
                keep_retrieving_run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state["thread"].id,
                    run_id=my_run.id
                )
                print(f"Run status: {keep_retrieving_run.status}")

                if keep_retrieving_run.status == "completed":
                    print("\n")

                    # Step 6: Retrieve the Messages added by the Assistant to the Thread
                    all_messages = client.beta.threads.messages.list(
                        thread_id=st.session_state["thread"].id
                    )

                    print("------------------------------------------------------------ \n")

                    print(f"User: {my_thread_message.content[0].text.value}")
                    print(f"Assistant: {all_messages.data[0].content[0].text.value}")

                    full_response = all_messages.data[0].content[0].text.value

                    break
                elif keep_retrieving_run.status == "queued" or keep_retrieving_run.status == "in_progress":
                    # sleep for 0.5 seconds
                    time.sleep(0.5)

                    pass
                else:
                    print(f"Run status: {keep_retrieving_run.status}")
                    break

            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Saving the chat history
        # chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
        # with open(chat_history_filename, "w") as file:
        #     file.write(json.dumps(st.session_state.messages))
