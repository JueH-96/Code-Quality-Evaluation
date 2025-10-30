# YOUR CODE HERE
import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# Use a generator expression to filter the string, keeping only '2' characters.
# Then, join these characters back into a single string.
# The generator expression iterates through S, yields `char` if `char` is '2',
# and "".join() concatenates the yielded characters.
result = "".join(char for char in S if char == '2')

# Print the resulting string to standard output
print(result)