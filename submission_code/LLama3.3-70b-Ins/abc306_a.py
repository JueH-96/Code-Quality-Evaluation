# Read the length of the string
N = int(input())

# Read the string
S = input()

# Initialize an empty string to store the result
result = ""

# Iterate over each character in the string
for char in S:
    # Append the character twice to the result
    result += char * 2

# Print the result
print(result)