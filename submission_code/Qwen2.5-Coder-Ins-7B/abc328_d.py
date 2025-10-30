# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Remove all occurrences of "ABC" from the string
while "ABC" in S:
    S = S.replace("ABC", "")

# Print the final string
print(S)