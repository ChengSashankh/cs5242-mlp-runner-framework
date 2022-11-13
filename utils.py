# Set up default conf
from base.constants import FILMS_GENRE


def get_default_sent_cleaner_conf():
    # Sentence cleaner conf
    sent_cleaner_conf = dict()
    sent_cleaner_conf['token'] = True
    sent_cleaner_conf['lower'] = True
    sent_cleaner_conf['encode'] = False
    sent_cleaner_conf['remove_stop'] = True
    sent_cleaner_conf['remove_punc'] = True
    sent_cleaner_conf['remove_esc'] = True
    sent_cleaner_conf['stem'] = False

    return sent_cleaner_conf


# Set up default conf
def get_default_down_sample_conf():
    # Down sample conf
    down_sample_conf = dict()

    for genre in FILMS_GENRE:
        down_sample_conf[genre] = 0

    return down_sample_conf
