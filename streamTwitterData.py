#! python3

import private
import re
import settings
import sys
import tweepy


# Define query stream search class
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        # Filter out retweets
        if status.retweeted or 'RT @' in status.text:
            return

        # Store data from all original tweets
        name = status.user.screen_name
        id_str = status.id_str
        time = str(status.created_at)
        tweet = re.sub(',', '', re.sub(r"http\S+", "", str(status.text))).replace('\n', '')

        print('{}, {}, {}, {}'.format(time, id_str, name, tweet))

    # Handle errors from Twitter API
    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            return False

# Set Twitter authorizations
auth = tweepy.OAuthHandler(private.cons_key, private.cons_sec)
auth.set_access_token(private.access_tok, private.access_sec_tok)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Test API
if not api:
    print("Can't Authenticate")
    sys.exit(-1)

# Run streamer
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(track=settings.searchQuery, async=True, languages=['en'])

