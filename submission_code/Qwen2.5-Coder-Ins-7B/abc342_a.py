# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Find the character that is different
for i in range(len(S)):
    if S[i] != S[0]:
        print(i + 1)
        break