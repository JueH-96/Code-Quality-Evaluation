# YOUR CODE HERE
import sys

# Read the input string from standard input
s = sys.stdin.readline().strip()

# Extract the first character of the string
first_char = s[0]

# Define the string to be appended
suffix = "UPC"

# Concatenate the first character and the suffix
result = first_char + suffix

# Print the result to standard output
print(result)