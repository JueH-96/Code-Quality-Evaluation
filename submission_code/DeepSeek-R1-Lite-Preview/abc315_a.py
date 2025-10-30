# Read the input string
S = input()

# Define the set of vowels
vowels = {'a', 'e', 'i', 'o', 'u'}

# Use list comprehension to collect non-vowel characters
result = [char for char in S if char not in vowels]

# Join the list into a string and print
print(''.join(result))