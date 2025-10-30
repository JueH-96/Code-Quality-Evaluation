# YOUR CODE HERE

import sys

def count_substrings(s):
    count = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            count += 1
    return count

s = sys.stdin.readline().strip()
print(count_substrings(s))