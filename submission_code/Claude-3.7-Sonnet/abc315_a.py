# Read the input string
S = input()

# Remove all vowels and join the remaining characters
result = ''.join(char for char in S if char not in "aeiou")

# Print the result
print(result)