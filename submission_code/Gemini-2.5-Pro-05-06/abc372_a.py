# Read the input string S
S = input()

# Remove all occurrences of '.' from S
# The replace() method returns a new string where all occurrences of a specified substring
# are replaced with another substring. Here, we replace '.' with an empty string '',
# effectively removing it.
result_string = S.replace('.', '')

# Print the resulting string
print(result_string)