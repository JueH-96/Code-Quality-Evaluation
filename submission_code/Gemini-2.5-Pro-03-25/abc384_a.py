# YOUR CODE HERE
import sys

# Read input from standard input
# The first line contains N, c1, c2 separated by spaces.
line1 = sys.stdin.readline().split()
# N = int(line1[0]) # The length N is provided but not explicitly needed in this Python solution logic
c1 = line1[1] # The character to keep
c2 = line1[2] # The character to replace others with

# The second line contains the string S.
# Use .strip() to remove any potential trailing newline character.
S = sys.stdin.readline().strip()

# Construct the result string
# We iterate through each character `char` in the input string `S`.
# We use a list comprehension to build a list of characters for the new string.
# - If the current character `char` is equal to `c1`, we keep `char`.
# - Otherwise (if `char` is not equal to `c1`), we use the character `c2`.
# Finally, "".join() concatenates the characters in the list into the final result string.
result_string = "".join([char if char == c1 else c2 for char in S])

# Print the final result string to standard output
print(result_string)