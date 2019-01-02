__author__ = 'Feng Chen'
import datetime
from time import gmtime, strftime, localtime
class Log(object):

    def __init__(self, filename = "log.txt"):
        self.tangerine = "And now a thousand years between"
        self.filename = filename
        # self.f = open(filename, 'a')
        str = strftime("%a, %d %b %Y %I:%M:%S %p %Z", localtime())
        self.write("\r\n\r\n--------------------------------------------------")
        self.write("--------------------------------------------------")
        self.write("\r\n-------------------" + str + "---------------------\r\n")
        # self.close()
        # datetime.datetime.time(datetime.datetime.now())

    def write(self, line):
        self.f = open(self.filename, 'a')
        self.f.write(line + "\r\n")
        print line
        self.f.close()

    def close(self):
        self.f.close()