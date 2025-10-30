# Read input values
N = int(input())
S = input()

# Initialize an empty string to store the result
result = ""

# Iterate through each character in the string S
for char in S:
    # Append the character twice to the result string
    result += char * 2

# Print the final result
print(result)