# Read the input string from stdin
S = input()

# Split the string by '.' and get the last part
parts = S.split('.')

# Print the last substring
print(parts[-1])