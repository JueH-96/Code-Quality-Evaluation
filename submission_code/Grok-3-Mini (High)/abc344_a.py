# Read the input string from standard input
S = input()

# Split the string by the '|' character
parts = S.split('|')

# Concatenate the first and third parts after splitting
result = parts[0] + parts[2]

# Print the resulting string
print(result)