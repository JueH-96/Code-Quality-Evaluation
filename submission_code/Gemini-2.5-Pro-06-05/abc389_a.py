# Read the input string from standard input
S = input()

# The first character of the string S is S[0]
# The third character of the string S is S[2]
# We need to convert these characters to integers to perform multiplication

# Convert the first character to an integer
a = int(S[0])

# Convert the third character to an integer
b = int(S[2])

# Calculate the product of the two integers
result = a * b

# Print the result to standard output
print(result)