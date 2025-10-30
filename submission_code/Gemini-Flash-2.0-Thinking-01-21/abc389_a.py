import sys

# Read the input string from stdin
s = sys.stdin.readline().strip()

# The string is guaranteed to be of the format "digit x digit"
# The first character is the first digit
digit1_char = s[0]
# The third character is the second digit
digit2_char = s[2]

# Convert the character digits to integers
digit1 = int(digit1_char)
digit2 = int(digit2_char)

# Calculate the product
product = digit1 * digit2

# Print the result
print(product)