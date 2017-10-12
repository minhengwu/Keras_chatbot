import os
from slackclient import SlackClient

bot_name = 'chatbot'

slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

if __name__ == '__main__':
    api_call = slack_client.api_call('users.list')
    if api_call.get('ok'):
        users = api_call.get('members')
        for user in users:
            if 'name' in user and user.get('name') == bot_name:
                print("Bot ID for '" + user['name'] + "' is " + user.get('id'))
    else:
        print("could not find bot user with the name " + bot_name)
