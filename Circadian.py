from utils.util import *
import streamlit as st
from PIL import Image
import os

pd.set_option('mode.chained_assignment',  None)

st.title("😴 Circadian Rhythm Algorithm 💤")
name = st.sidebar.selectbox('Menu', ['📄 CSS'])

form = st.form("my_form")
file = form.file_uploader("Upload Data", type=["csv", "xlsx"])

if os.path.exists('graph.png'):
    os.remove('graph.png')

if file is not None:
    #origin_image = pd.read_csv(image_file)
    df = pd.read_csv(file)
    sleep_df = df[['sleep_start_day', 'sleep_start_time', 'sleep_end', 'sleep_end_time']]
    sleep_light_df = make_sleep_light(sleep_df)
    sleep_light_df.columns = [0,1]
    waso_df = df[['waso', 'main']]
    waso_df.columns = ['0', '1']
    #sleep_light_df = pd.read_csv('./data/Input1_sleep_light3.csv', header=None)
    #waso_df = pd.read_csv('./data/Input2_WASO_main3.csv', header=None)


submitted = form.form_submit_button("수면 분석을 시작합니다.")

if submitted:
    fn = 'graph.png'
    CSS, ness_sleep_amount = main(sleep_light_df, waso_df)
    if CSS != -100:
        if CSS == 0.0:
            is_sufficient = 'Insufficient'
        elif CSS == 1.0:
            is_sufficient = 'Sufficient'
        elif CSS == -99:
            is_sufficient = 'Not Available'
        
        if ness_sleep_amount == -99:
            ns = 'Not Available'
        else:
            ns = str(np.round(ness_sleep_amount, 1)) + ' Hours'

        image = Image.open(fn)
        st.image(image, caption='Circadian Rhythm Graph')
        
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )

        st.subheader('Sleep Satisfaction')
        st.text(is_sufficient)
        st.subheader('Necessary Sleep Amount')
        st.text(ns)
    else:
        st.subheader('Failed to get algorithm results')


