# Read the input from stdin
S = input()

# Find the last substring after the last dot
last_substring = S.split('.')[-1]

# Print the last substring
print(last_substring)