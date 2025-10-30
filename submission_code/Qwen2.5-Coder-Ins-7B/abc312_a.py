# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Define the valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input string is in the set of valid strings
if S in valid_strings:
    print("Yes")
else:
    print("No")