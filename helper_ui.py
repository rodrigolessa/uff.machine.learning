import os
import sys

class UIHelper:

  def __init__(self, total):
    self.interator = 0
    self.total = total

  def progress(self, status=''):
    self.interator+=1

    bar_len = 60
    filled_len = int(round(bar_len * self.interator / float(self.total)))

    percents = round(100.0 * self.interator / float(self.total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()