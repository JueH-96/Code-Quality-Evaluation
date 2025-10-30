# YOUR CODE HERE
# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input string is in the set of valid strings
if input in valid_strings:
    print("Yes")
else:
    print("No")