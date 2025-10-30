# Read the input string from stdin
S = input()

# Remove all vowels (a, e, i, o, u) and create the resulting string
result = ''.join(c for c in S if c not in 'aeiou')

# Print the resulting string to stdout
print(result)