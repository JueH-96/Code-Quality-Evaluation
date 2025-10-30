# YOUR CODE HERE
import sys

S = sys.stdin.readline().strip()

lower_count = sum(1 for c in S if c.islower())
upper_count = sum(1 for c in S if c.isupper())

if upper_count > lower_count:
    S = S.upper()
else:
    S = S.lower()

print(S)