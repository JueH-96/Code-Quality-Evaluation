# Read the input string
S = input().strip()

# Define the vowels to remove
vowels = {'a', 'e', 'i', 'o', 'u'}

# Filter out vowels and join the remaining characters
result = ''.join([char for char in S if char not in vowels])

# Print the resulting string
print(result)