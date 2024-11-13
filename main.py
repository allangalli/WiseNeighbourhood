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

def main():
    """Main function to initialize and run the Streamlit application."""
    initialize_session()
    handle_sidebar()

    if "expander_state" not in st.session_state:
        st.session_state["expander_state"] = True

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
            st.caption("If you don't know your neighbourhood, you can look it up here: [Find Your Neighbourhood](https://www.toronto.ca/city-government/data-research-maps/neighbourhoods-communities/neighbourhood-profiles/find-your-neighbourhood/#location=&lat=&lng=&zoom=)") 
            st.session_state.input_text = neighbourhood
        else:
            print(len(st.session_state.messages))
            print(st.session_state.messages)

        col1, col2 = chat_container.columns(2)

        if col1.button("Submit", type="primary", use_container_width=True):
            # Check if the input text is not empty
            if len(st.session_state.messages) >= 6:
                st.write("## Placeholder: LLM output after intake information pass")
            elif len(st.session_state.messages) != 0 or st.session_state.input_text.strip():
                handle_submission()
            else:
                st.warning("Please select a neighbourhood before submitting")

        if col2.button("Restart Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.user_inputs = {}
            st.session_state.input_text = ''
            st.rerun()


if __name__ == "__main__":
    main()
