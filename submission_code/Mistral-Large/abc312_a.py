import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if S is in the set of valid strings
if S in valid_strings:
    print("Yes")
else:
    print("No")