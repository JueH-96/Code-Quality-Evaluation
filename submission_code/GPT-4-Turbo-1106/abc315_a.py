# Read the input string from standard input
S = input().strip()

# Define the vowels to be removed
vowels = 'aeiou'

# Remove all occurrences of vowels from the string
result = ''.join(char for char in S if char not in vowels)

# Print the resulting string
print(result)