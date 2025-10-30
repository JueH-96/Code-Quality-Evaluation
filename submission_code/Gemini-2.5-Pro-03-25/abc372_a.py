import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# Use the string replace method to remove all occurrences of '.'
# The second argument '' means replace '.' with an empty string, effectively removing it.
result_string = S.replace('.', '')

# Print the resulting string to standard output
print(result_string)