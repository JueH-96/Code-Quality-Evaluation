# Read the input string
S = input().strip()

# Count the number of uppercase and lowercase letters
uppercase_count = sum(1 for char in S if char.isupper())
lowercase_count = sum(1 for char in S if char.islower())

# Determine which conversion to apply
if uppercase_count > lowercase_count:
    # Convert all lowercase to uppercase
    S = S.upper()
else:
    # Convert all uppercase to lowercase
    S = S.lower()

# Print the result
print(S)