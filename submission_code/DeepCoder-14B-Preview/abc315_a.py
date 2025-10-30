# Read the input string
s = input().strip()

# Remove all vowels and construct the result string
result = ''.join([c for c in s if c not in {'a', 'e', 'i', 'o', 'u'}])

# Print the resulting string
print(result)