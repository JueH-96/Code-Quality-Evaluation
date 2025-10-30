# Read the input string from Standard Input
S = input()

# Define the vowels
vowels = "aeiou"

# Use a list comprehension to build a list of characters from S
# that are not in the 'vowels' string.
non_vowel_chars = [char for char in S if char not in vowels]

# Join the characters in the list to form the result string.
# If non_vowel_chars is empty (e.g. S="aei"), result_string would be "",
# but the problem constraints state S contains at least one non-vowel.
result_string = "".join(non_vowel_chars)

# Print the result string to Standard Output
print(result_string)