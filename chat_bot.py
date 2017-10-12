import os
import time
from slackclient import SlackClient
from lisa import Bot
import pickle
import numpy as np
from train import to_cat

# starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "do"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))


def handle_command(bot,command, channel):
    #command will go through a neural net work and product a response
    input_seq = word2vec(command)
    response = bot.analyze(input_seq)
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)

def word2vec(command):
    command = [word_to_idx.get(i) if word_to_idx.get(i) else 2 for i in command.split()]
    command.extend([0] * (20- len(command)))
    command = to_cat(np.array(command))
    command = np.reshape(command, newshape=[1,20,5004])
    # return command
    return command


def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and output['user'] != BOT_ID:
                # return text after the @ mention, whitespace removed
                return output['text'], output['channel']
    return None, None



if __name__ == "__main__":
    with open('word_to_idx.pickle', 'rb') as handle:
        word_to_idx = pickle.load(handle)
        handle.close()
    with open('idx_to_word.pickle', 'rb') as handle:
        idx_to_word = pickle.load(handle)
        handle.close()
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    bot = Bot('my_model_weights.h5', 'word_to_idx.pickle','idx_to_word.pickle')
    if slack_client.rtm_connect():
        print("Hello. How are you feeling today?")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(bot,command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
