import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# Split the string by '.' and get the last element
last_substring = S.split('.')[-1]

# Print the last substring
print(last_substring)