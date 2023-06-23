import os
from slack_bolt import App
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from slack_bolt.adapter.socket_mode import SocketModeHandler

# create a new Slack app with your bot token
app = App(token="xoxb-4982809354582-5428728496433-slCdKrFcQXa17zlD96QNp6S1")
SocketModeHandler(app, "xapp-1-A05CXP95LGY-5416070689443-8d0357efdd00579a5cfd1854b605ac35f21eb295ac3863dbc58546b933e0020b").start()

# load FAQs into a DataFrame
df = pd.read_csv('faqs.csv')

# preprocess the text
df['Question'] = df['Question'].apply(
  lambda x: ' '.join(x.lower() for x in x.split()))
df['Answer'] = df['Answer'].apply(
  lambda x: ' '.join(x.lower() for x in x.split()))

# create a TF-IDF vectorizer
tfidf = TfidfVectorizer()

# fit the vectorizer on the questions
X = tfidf.fit_transform(df['Question'])

# create a Naive Bayes classifier
nb = MultinomialNB()

# train the classifier on the questions and answers
nb.fit(X, df['Answer'])

# define a function to get the chatbot's response
def get_response(question):
  question_tfidf = tfidf.transform([question])
  answer = nb.predict(question_tfidf)
  return answer[0]

# define an event listener for incoming messages
@app.event("app_mention")
def handle_app_mention_events(body, logger):
    logger.info(body)
    print(body)
  
# define an event listener for incoming messages
@app.event("message")
def handle_message(event, say):
  
# get the user's message text
message_text = event["text"]
  
# get the chatbot's response
response_text = get_response(message_text)

# send the response back to the user
  try:
    say(response_text)
  except SlackApiError as e:  
    print(f"Error sending message: {e}")
    
# start the app
if __name__ == "__main__":
  app.start(port=int(os.environ.get("PORT", 3000)))
