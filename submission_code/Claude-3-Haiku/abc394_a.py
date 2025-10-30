# Read the input from stdin
S = input()

# Create a new string by concatenating all the '2' characters from the input string
result = ''.join(char for char in S if char == '2')

# Print the result to stdout
print(result)