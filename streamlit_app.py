import streamlit as st
from SmartApi import SmartConnect  # Import the SmartApi library
import pyotp
from logzero import logger

# Set page configuration
st.set_page_config(page_title="Donzai Fincorp Group", layout="wide", initial_sidebar_state="collapsed")

# Check if user is logged in
if "authToken" in st.session_state:
    st.switch_page("pages/1_Dashboard.py")

# Custom CSS for loading overlay
st.markdown("""
    <style>
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        z-index: 9999;
        display: none;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        color: black;
    }
    .loading-overlay.show {
        display: flex;
    }
    </style>
""", unsafe_allow_html=True)

# # Initialize session state for loading overlay
if "logging_Error" not in st.session_state:
    st.session_state.logging_Error = None

# Use columns for side-by-side layout
col1, col2, col3 = st.columns([1, 1, 2])  # Adjust ratios as needed
with col1:
    st.markdown("<h1 style='font-size: 35px; margin-bottom: 0;'>Quant by DFG</h1>", unsafe_allow_html=True)  # Adjust font size
with col2:
    st.markdown("<h2 style='font-size: 15px; margin-top: 22px;'>Smart API by AngelOne</h2>", unsafe_allow_html=True)  # Adjust font size and spacing

# Login Form
st.write("##### Login Credentials")
with st.form(key="login_form"):
    api_key = st.text_input("API Key", placeholder="Your API Key")
    username = st.text_input("AngelOne Username", placeholder="Your AngelOne Username")
    pwd = st.text_input("AngelOne Login PIN", placeholder="Your AngelOne Login PIN", type="password")
    token = st.text_input("TOTP Token", placeholder="Your TOTP Token")
    login_button = st.form_submit_button("Login")
    

# Authentication Logic as a Method
def authenticate_user():
    if not api_key or not username or not pwd or not token:
        st.error("All fields are required!")
        return
    else:
        # Show loading overlay
        st.markdown('<div class="loading-overlay show">Authenticating...</div>', unsafe_allow_html=True)

        with st.spinner("Authenticating..."):
            try:
                # Initialize SmartConnect
                smartApi = SmartConnect(api_key=api_key)

                # Generate TOTP using pyotp
                totp = pyotp.TOTP(token).now()

                # Call generateSession
                data = smartApi.generateSession(username, pwd, totp)

                if data["status"] is False:
                    error_message = f"Login Failed: {data.get('message', 'Unknown Error')}"
                    st.session_state.logging_Error = error_message
                    st.error(error_message)
                    st.rerun()
                else:
                    # On Success
                    authToken = data['data']['jwtToken']
                    refreshToken = data['data']['refreshToken']
                    feedToken = smartApi.getfeedToken()
                    profile_data = smartApi.getProfile(refreshToken)

                    st.success("Login Successful!")

                    # Store data in session state
                    st.session_state.api_key = api_key
                    st.session_state.username = username
                    st.session_state.totpToken = token

                    st.session_state.authToken = authToken
                    st.session_state.refreshToken = refreshToken
                    st.session_state.feedToken = feedToken
                    st.session_state.profile_data = profile_data
                    st.session_state.logging_Error = None

                    # Navigate to Dashboard
                    st.switch_page("pages/1_Dashboard.py")

            except Exception as e:
                logger.error(e)
                error_message = f"An error occurred during authentication: {e}"
                st.error(error_message)
                st.session_state.logging_Error = error_message
                st.rerun()

# Display the error message if exists
if st.session_state.logging_Error:
    st.error(st.session_state.logging_Error)

# Trigger the authentication method when login_button is clicked
if login_button:
    authenticate_user()