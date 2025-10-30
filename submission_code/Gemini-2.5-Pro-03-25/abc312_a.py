# YOUR CODE HERE
import sys 

# Read the input string from standard input
# input() reads a line from stdin, removes the trailing newline, and returns it as a string.
s = input()

# Define the set of target strings. 
# Using a set provides efficient membership checking (using the 'in' operator).
target_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input string 's' is present in the set 'target_strings'.
if s in target_strings:
    # If 's' is found in the set, it means it matches one of the required strings.
    # Print "Yes" to standard output.
    print("Yes")
else:
    # If 's' is not found in the set, it does not match any of the required strings.
    # Print "No" to standard output.
    print("No")