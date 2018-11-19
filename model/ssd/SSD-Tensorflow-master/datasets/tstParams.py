import sys
from optparse import OptionParser
def add (one,two):
    print (one+two)

if __name__=='__main__':
    one = sys.argv[0]
    one = str(one)
    params = one.strip().split(",")
    print (params[0])
    print (params[1])
    add(int(params[0]),int(params[1]))