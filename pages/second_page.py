import streamlit as st
import time
import replicate

st.set_page_config(page_title="Company 2", page_icon="📈")

st.write("Welcome to main")

with st.sidebar: 
    st.button("Add new casesheet", type="secondary")
    #st.write("Patient 1")
    #with st.container():
    #    st.write("This is inside the container")
    st.code(''' 
                18 Apr, 2023
                Dermatitis to parthenium
                Complaints of itchy oozy lesions over 
                body for 3 years, flared for 2 weeks...
                Lab results pending
            ''', 
            None)
    st.code(''' 
                18 Apr, 2023
                Dermatitis to parthenium
                Complaints of itchy oozy lesions over 
                body for 3 years, flared for 2 weeks...
                Lab results pending
            ''', 
            None)
    


st.markdown(
    """
    Patient History > Casesheet ID
    """
)

st.divider()

st.markdown(
    """
    Allergic contact dermatitis to parthenium
    Treating Doctor   Dr Rashmi Kumari
    Admission Date    18 April 2023, 10:30 am
    Discharged on     28 April 2023, 12:06 pm
    Status            Lab results pending
    """
)

tab1, tab2, tab3 = st.tabs(["Present Illness", "Examination", "Procedure Plan"])


with tab1: 
    st.header("Present Illness")
    txt = st.text_area('Text to analyze', '''It was the best me time''')
    button_submit = st.button("Submit", type="primary")
    if button_submit:
        st.write("Thanks")
        with st.spinner("Loading..."):
            time.sleep(5)
        st.success("Done!")