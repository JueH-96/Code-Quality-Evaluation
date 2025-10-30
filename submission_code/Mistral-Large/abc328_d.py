import sys
import re

# Read input from stdin
S = sys.stdin.read().strip()

# Use regular expression to replace all occurrences of 'ABC'
while 'ABC' in S:
    S = S.replace('ABC', '', 1)

# Print the final string
print(S)