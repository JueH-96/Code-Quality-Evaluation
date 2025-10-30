# YOUR CODE HERE
import sys

S = sys.stdin.read().strip()

uppercase_count = sum(1 for c in S if c.isupper())
lowercase_count = sum(1 for c in S if c.islower())

if uppercase_count > lowercase_count:
    result = S.upper()
else:
    result = S.lower()

print(result)