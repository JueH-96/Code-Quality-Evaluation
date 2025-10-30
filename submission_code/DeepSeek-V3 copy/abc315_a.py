# Read the input string
S = input().strip()
# Define the vowels to be removed
vowels = {'a', 'e', 'i', 'o', 'u'}
# Remove all vowels from the string
result = ''.join([char for char in S if char not in vowels])
# Print the result
print(result)