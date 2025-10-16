import streamlit as st

st.title("Мій перший Streamlit застосунок 🚀")

st.write("Це приклад застосунку, створеного прямо з GitHub!")

name = st.text_input("Введи своє ім’я:")
if name:
    st.success(f"Привіт, {name}! 👋")
