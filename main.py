"""Streamlit app module for interactive chat management and display."""
from typing import Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from llm_utils.conversation import Conversation
from llm_utils.prompt_assembly import prompt_assembly
from llm_utils.stream_handler import StreamUntilSpecialTokenHandler
from streamlit_utils.initialization import initialize_session
from streamlit_utils.ui_creator import display_ui_from_response
from llm_utils.maps import neighbourhood_select
import uuid
import requests
import time
import threading
import random
import json

# Page configuration
st.set_page_config(
    page_title="TPS Safety Management Planner",
    page_icon="👮‍♀️",
    layout="wide",
    initial_sidebar_state="expanded")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
            
[data-testid="stImage"]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

# Function to send plan request to Node.js server
def send_plan_request(inputData, id):
    # Replace 'your-nodejs-server-address' with your actual server address or IP
    url = 'http://15.222.45.62:5200/plan'
    payload = {
        'inputData': inputData,
        'id': id
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending plan request: {e}")
        return None

# Function to get plan from Node.js server
def get_plan(id):
    url = 'http://15.222.45.62:5200/getPlan'
    payload = {'id': id}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving plan: {e}")
        return None


def get_survey_respond(info):
    # Initialize the response dictionary
    response = {
        "Neighbourhood": "",
        "Crime Type": [],
        "user-context": []
    }
    
    # Parse the first message to get neighbourhood and crime data
    first_message = info[0].content
    if "Neighbourhood:" in first_message:
        # Extract neighbourhood name and ID
        neighbourhood_part = first_message.split(" - ")[0]
        response["Neighbourhood"] = neighbourhood_part.replace("Neighbourhood: ", "")
        
        # Extract crime types and levels
        crime_part = first_message.split(" - ")[1]
        crime_items = crime_part.split(", ")
        crime_data = []
        for item in crime_items:
            crime_type, level = item.strip().split(": ")
            if level.rstrip(";"):  # Remove trailing semicolon if present
                crime_data.append(f"{crime_type}: {level}")
        response["Crime Type"] = crime_data
    
    # Parse user response (third message in the list) and AI response (fourth message)
    if len(info) >= 4:
        user_response = info[2].content
        ai_response = info[3].content
        
        # Extract Q&A pairs
        user_selections = user_response.strip().split(";\n")
        for selection in user_selections:
            if selection:  # Skip empty strings
                q, a = selection.split(": ")
                response["user-context"].extend([f"Q: {q}", f"A: {a.rstrip(';')}"]) 

    return response

def get_conversation() -> Optional[Conversation]:
    """Retrieve the current conversation instance from Streamlit's session state."""
    return st.session_state.get("conversation", None)

def neighbourhoods():
    nb = './Assets/Safety Risks by Neighbourhood & Offence.csv'
    regions = pd.read_csv(nb)
    return regions['Neighbourhood']

def get_offence_risk(region):
    nb = './Assets/Safety Risks by Neighbourhood & Offence.csv'
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(nb)
    
    # Filter the DataFrame for the specified region
    df_region = df[df['Neighbourhood'] == region]
    
    # Create a list of offences and their corresponding safety risks
    offences_list = df_region[['Offence Group', 'Safety Risk']].values.tolist()
    
    # Format the offences and risks into a string
    offences_str = ', '.join([f"{offence}: {risk}" for offence, risk in offences_list])
    
    # Construct the final result string
    result = f"Neighbourhood: {region} - {offences_str}"
    return result


def handle_submission():
    """Process and submit user input, updating conversation history."""
    user_input = st.session_state.input_text
    user_prompt = prompt_assembly(st.session_state.user_inputs, user_input)
    user_message = HumanMessage(role="user", content=user_prompt)
    st.session_state.messages.append(user_message)
    st.session_state.conv_history.append(user_message)

    conversation_instance = get_conversation()

    with st.chat_message("assistant"):
        stream_handler = StreamUntilSpecialTokenHandler(st.empty())

        textual_response, json_response = conversation_instance(
            user_message, stream_handler)

        st.session_state.conv_history.append(AIMessage(
            role="assistant", content=textual_response))
        st.session_state.conv_history.append(
            AIMessage(role="assistant", content=json_response))
        st.session_state.messages.append(AIMessage(
            role="assistant", content=json_response))

    st.session_state.input_text = ""

    st.session_state.user_inputs = {}
    st.rerun()


def handle_sidebar():
    """Manage sidebar interactions for model selection and updates in the Streamlit app."""
    with st.sidebar:
        st.subheader("Conversation Agent")
        conv_selection = model_selection("Conversation")
        st.subheader("UI Agent")
        ui_selection = model_selection("UI")

        if st.button("Update Agents"):
            update_conversation(conv_selection, ui_selection)


def update_conversation(conv_selection, ui_selection):
    """Update the conversation instance with new models for conversation and UI agents."""
    conversation_instance = get_conversation()
    conversation_instance.update_agents(conv_selection, ui_selection)


def model_selection(agent):
    """Display and handle model selection radio button."""
    label = f"{agent} Model"
    options = st.session_state[f"supp_models_{agent.lower()}"]
    index = options.index(st.session_state[f"sel_model_{agent.lower()}"])

    selection = st.radio(label, options, index)
    st.session_state[f"sel_model_{agent.lower()}"] = selection

    return selection

# This is the function to submit all the information to the database
def final_submission():
    # Generate a unique ID if not already generated
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())

    # Prepare the input data
    respond = get_survey_respond(st.session_state.messages)
    inputData = json.dumps(respond)
    id = st.session_state["user_id"]

    # Function to send plan request in a separate thread
    def send_request():
        # Send the plan request to the Node.js server
        send_response = send_plan_request(inputData, id)
        st.session_state['send_response'] = send_response

    # Start the progress bar immediately
    progress_bar = st.progress(0)
    status_text = st.empty()
    #st.success("Your plan is start to generating.")
    
    # Display a message about the expected time
    st.write("⏳ This process might take some time, up to 2 minutes.")

    # Start the send_request function in a separate thread
    send_thread = threading.Thread(target=send_request)
    send_thread.start()

    second = random.randint(70,90)
    # Display the progress bar for 1.5 minutes (90 seconds)
    for percent_complete in range(second):
        time.sleep(1)  # Sleep for 1 second
        progress = (percent_complete + 1) / second
        progress_bar.progress(progress)
        status_text.text(f"Generating your plan... {int(progress * 100)}%")

    # Wait for the send_request thread to finish
    send_thread.join()

    plan_response = get_plan(id)
    if plan_response and plan_response.get('success'):
        plan_text = plan_response.get('plan')
        # Display the plan to the user
        st.session_state.plan_text = plan_text
        st.write("## Your Safety Plan:")
        st.write(plan_text)
        st.session_state.plan_displayed = True
    else:
        st.error("Failed to retrieve the plan.")


def main():
    """Main function to initialize and run the Streamlit application."""
    initialize_session()
    ##handle_sidebar()

    if "expander_state" not in st.session_state:
        st.session_state["expander_state"] = True

     # Initialize session state variables
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    if 'plan_displayed' not in st.session_state:
        st.session_state.plan_displayed = False

    # Dashboard Main Panel
    col = st.columns((1, 4.5, 1), gap='medium')

    with col[1]:
        left_co, cent_co,last_co = st.columns((1.5, 4.5, 2))
        with cent_co:
            st.image("./Assets/TPS_Logo.png", width=120)

        st.markdown("<h1 style='display: flex; text-align: center;'>SixSafety - Your Neighbourhood Safety Advisor</h1>", unsafe_allow_html=True)
        # Welcome message
        st.write("## Welcome! Please let us know what area of your neighbourhood safety you'd like to learn more about.",)
        
        chat_container = st.container()

        # Display conversation history
        for index, msg in enumerate(st.session_state.messages):
            if msg.role == "assistant":
                with chat_container.chat_message("assistant"):
                    display_ui_from_response(
                        msg.content, index, len(st.session_state.messages) - 1)
            else:
                chat_container.chat_message(msg.role).write(msg.content)

        if len(st.session_state.messages) == 0:
            neighbourhood = st.selectbox(
                'Choose a Neighbourhood',
                neighbourhoods().sort_values().unique().tolist(),
                index=0,
                placeholder='start typing...',
            )
            intake_output = get_offence_risk(neighbourhood)
            ##print(intake_output)
            st.caption("If you don't know your neighbourhood, you can look it up here: [Find Your Neighbourhood](https://www.toronto.ca/city-government/data-research-maps/neighbourhoods-communities/neighbourhood-profiles/find-your-neighbourhood/#location=&lat=&lng=&zoom=)") 
            st.session_state.input_text = intake_output
        else:
            print(len(st.session_state.messages))
            ##print(st.session_state.messages)

        col1, col2 = chat_container.columns(2)

        if not st.session_state.submitted:
            if col1.button("Submit", type="primary", use_container_width=True):
                # Check if the input text is not empty
                if len(st.session_state.messages) >= 6:
                    st.session_state.submitted = True  # Set submitted to True
                    final_submission()
                elif len(st.session_state.messages) != 0 or st.session_state.input_text.strip():
                    handle_submission()
                else:
                    st.warning("Please select a neighbourhood before submitting")

        # Display the Restart Session button if:
        # - The plan has been displayed, or
        # - The user has not submitted yet (before clicking Submit)
        if st.session_state.plan_displayed or not st.session_state.submitted:
            if col2.button("Restart Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.user_inputs = {}
                st.session_state.input_text = ''
                st.session_state.submitted = False
                st.session_state.plan_displayed = False
                st.rerun()


if __name__ == "__main__":
    main()
