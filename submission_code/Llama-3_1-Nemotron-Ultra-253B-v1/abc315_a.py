# Read the input string
S = input().strip()

# Remove vowels by checking each character
result = ''.join([c for c in S if c not in {'a', 'e', 'i', 'o', 'u'}])

# Print the result
print(result)