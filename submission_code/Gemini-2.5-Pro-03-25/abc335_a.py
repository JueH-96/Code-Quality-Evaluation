# YOUR CODE HERE
import sys

# Read input from stdin
# input() reads a single line from standard input and returns it as a string.
# .strip() removes leading/trailing whitespace, although for this problem it might not be strictly necessary
# as the problem statement implies the input string has no extra whitespace.
S = sys.stdin.readline().strip()

# Python strings are immutable. To change a character, we need to create a new string.
# S[:-1] creates a slice of the string S containing all characters except the last one.
# We then concatenate this slice with the character '4'.
modified_S = S[:-1] + '4'

# Print the modified string to standard output.
print(modified_S)