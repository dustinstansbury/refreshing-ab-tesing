# Refreshing AB Testing
Streamlit app for running AB test analyses powered by [spearmint](https://github.com/dustinstansbury/spearmint)

## Try it on Streamlit Cloud [![Try it on Streamlit](https://static.streamlit.io/badges/streamlit_badge_red.svg)](https://ab-testing.streamlit.app)

## Use the app locally

```bash
# Clone the repo
# $ https://github.com/dustinstansbury/refreshing-ab-testing.git  # clone via https
$ git@github.com:dustinstansbury/refreshing-ab-testing.git  # clone repo ssh

# Create and activate virtualenv in the repo
$ cd refreshing-ab-testing
$ python3 -m venv .venv
$ source .venv/bin/activate

# Upgrade venv pip and install app requirements
$ python3 -m pip install --upgrade pip
$ pip install -r requirements.txt

# Run the app
$ streamlit run Hypothesis_Test.py
```