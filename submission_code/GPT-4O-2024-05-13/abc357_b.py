# YOUR CODE HERE
import sys
input = sys.stdin.read().strip()

S = input
lowercase_count = sum(1 for c in S if c.islower())
uppercase_count = len(S) - lowercase_count

if uppercase_count > lowercase_count:
    print(S.upper())
else:
    print(S.lower())