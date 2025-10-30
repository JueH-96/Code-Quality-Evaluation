import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# Filter the string to keep only the characters that are '2'
# A list comprehension is used to create a list of '2's
filtered_chars = [char for char in S if char == '2']

# Join the list of characters back into a single string
result = "".join(filtered_chars)

# Print the resulting string to standard output
print(result)