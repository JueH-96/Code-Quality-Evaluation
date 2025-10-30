import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# The string S is guaranteed to be in the format digit x digit
# Extract the first character (the first digit)
digit1_char = S[0]

# Extract the third character (the second digit)
digit2_char = S[2]

# Convert the character digits to integers
digit1 = int(digit1_char)
digit2 = int(digit2_char)

# Calculate the product of the two digits
product = digit1 * digit2

# Print the result
print(product)