# YOUR CODE HERE

import sys

S = sys.stdin.readline().strip()

upper_count = sum(1 for c in S if c.isupper())
lower_count = sum(1 for c in S if c.islower())

if upper_count > lower_count:
    print(S.lower())
else:
    print(S.upper())