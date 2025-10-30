# YOUR CODE HERE

import sys

def find_diff_char(s):
    for i in range(1, len(s)):
        if s[i] != s[0]:
            return i+1
    return len(s)

s = sys.stdin.readline().strip()
print(find_diff_char(s))