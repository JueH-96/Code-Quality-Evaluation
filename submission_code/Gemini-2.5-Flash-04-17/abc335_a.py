import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# The string S is guaranteed to end with "2023".
# This means the last character is '3'.
# To change the last character to '4', we can take a slice of the string
# up to the second-to-last character (S[:-1]) and append '4'.
modified_S = S[:-1] + '4'

# Print the modified string to stdout
print(modified_S)