# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Check if the first character is uppercase and all other characters are lowercase
if S[0].isupper() and S[1:].islower():
    print("Yes")
else:
    print("No")