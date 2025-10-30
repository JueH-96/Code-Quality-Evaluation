# YOUR CODE HERE

import sys

def remove_substring(s):
    while 'ABC' in s:
        s = s.replace('ABC', '', 1)
    return s

s = sys.stdin.readline().strip()
print(remove_substring(s))