# Read the input string S from Standard Input
S = input()

# Strings in Python are immutable. To change a character,
# we need to create a new string.
# S[:-1] gives a slice of the string S containing all characters
# except the last one.
# We then concatenate this slice with the character '4'.
modified_S = S[:-1] + '4'

# Print the modified string to Standard Output
print(modified_S)