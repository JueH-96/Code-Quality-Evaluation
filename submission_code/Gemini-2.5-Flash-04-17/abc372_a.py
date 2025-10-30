import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# Remove all occurrences of '.' from the string
# The replace method replaces all occurrences of the first argument with the second argument.
# Replacing '.' with an empty string effectively removes them.
result = S.replace('.', '')

# Print the resulting string to standard output
print(result)