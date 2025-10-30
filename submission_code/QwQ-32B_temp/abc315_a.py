# Read the input string
S = input().strip()

# Filter out the vowels and join the remaining characters
result = ''.join([c for c in S if c not in {'a', 'e', 'i', 'o', 'u'}])

# Print the resulting string
print(result)