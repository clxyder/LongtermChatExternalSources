RAVEN_NAME = "RAVEN"
USER_NAME = "USER"

# API Keys
CONFIG_DEFAULT_KEY = "DEFAULT"
CONFIG_OPENAI_API_KEY = "OPENAI_API_KEY"

# File System
CHAT_LOG_DIR = "chat_logs"

# Prompt Keys
INPUT_KEY = "<<INPUT>>"
NOTES_KEY = "<<NOTES>>"
CONVERSATION_KEY = "<<CONVERSATION>>"

# Prompt Templates
PROMPT_NOTES_TEMPLATE = f'''Write detailed notes of the following in a hyphenated list format like "- "



{INPUT_KEY}



NOTES:'''

PROMPT_RESPONSE_TEMPLATE = f'''I am a chatbot named RAVEN. My goals are to reduce suffering, increase prosperity, and increase understanding. I will read the conversation notes and recent messages, and then I will provide a long, verbose, detailed answer.



The following are notes from earlier conversations with USER:
{NOTES_KEY}



The following are the most recent messages in the conversation:
{CONVERSATION_KEY}



I will now provide a long, detailed, verbose response:
{RAVEN_NAME}:'''