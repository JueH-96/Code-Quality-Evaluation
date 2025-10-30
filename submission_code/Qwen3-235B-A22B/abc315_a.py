# Read the input string
S = input().strip()

# Define the vowels to be removed
vowels = {'a', 'e', 'i', 'o', 'u'}

# Filter out the vowels and join the remaining characters
result = ''.join([char for char in S if char not in vowels])

# Print the resulting string
print(result)