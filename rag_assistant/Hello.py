import streamlit as st
import sys
from utils.config_loader import load_config

config = load_config()

app_name = config['DEFAULT']['APP_NAME']

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


def main():
    st.title("Welcome Page")

    st.write(f"""# Welcome to {app_name} ! 👋""")

    st.markdown(
        """
        Enhance data accuracy and consistency for Engie entities by automating, encoding and updates via GenIA, centralized in DiliTrust, a single source of truth for legal entity data. ​

        Search specific legal information on entities in full autonomy, using natural language.
    """
    )
    st.write(sys.version)


if __name__ == "__main__":
    main()
