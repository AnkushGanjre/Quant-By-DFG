import streamlit as st

# Check if user is logged in
if "authToken" not in st.session_state:
    st.warning("Please log in to access the Dashboard.")
    st.switch_page("streamlit_app.py")

# Display dashboard content
st.title("Dashboard")
st.write("Welcome to Algo by DFG!")

# Display user profile data
st.write("Profile Data:", st.session_state.profile_data)