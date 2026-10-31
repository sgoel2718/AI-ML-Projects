"""
Text cleaning.

Raw messages are messy: capital letters, links, @usernames, HTML tags,
punctuation and emojis. A model treats "FREE", "Free" and "free!!!" as three
different words unless we clean them into one form first.

Try it:
    >>> clean_text("FREE entry!!! Visit http://win.com now @john")
    'free entry visit url now user'
"""

import re

# Each pattern is compiled once here, which is faster than re-reading it for
# every one of the 1.6 million messages.
LINK = re.compile(r"https?://\S+|www\.\S+")  # http://..., https://..., www....
USERNAME = re.compile(r"@\w+")                # @john, @team_lead
HTML_TAG = re.compile(r"<[^>]+>")             # <br>, <p class="x">
NOT_A_LETTER = re.compile(r"[^a-z' ]+")       # anything except a-z, ' and space
EXTRA_SPACES = re.compile(r"\s+")             # two or more spaces in a row


def clean_text(text):
    """Return a lowercase, letters-only version of one message."""
    text = str(text).lower()               # "FREE Prize" -> "free prize"
    text = LINK.sub(" url ", text)         # every link becomes the word "url"
    text = USERNAME.sub(" user ", text)    # every @name becomes the word "user"
    text = HTML_TAG.sub(" ", text)         # remove HTML tags from emails
    text = NOT_A_LETTER.sub(" ", text)     # remove digits, punctuation, emojis
    text = EXTRA_SPACES.sub(" ", text)     # "free   prize" -> "free prize"
    return text.strip()                    # remove spaces at the start and end

# Why keep "url" and "user" instead of deleting links and names?
# Because "this message contains a link" is itself a strong spam signal,
# while the exact link (which changes every time) is not.
#
# Why keep apostrophes? So "don't" and "can't" stay single words. Unlike the
# Day 3 notebook, we also keep small words like "not" and "no": in sentiment,
# "good" and "not good" mean opposite things.
