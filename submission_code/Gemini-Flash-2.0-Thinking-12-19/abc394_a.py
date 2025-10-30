import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# Filter the string to keep only the character '2'
# This can be done using a list comprehension or a generator expression
filtered_string = "".join(char for char in S if char == '2')

# Print the resulting string to stdout
print(filtered_string)