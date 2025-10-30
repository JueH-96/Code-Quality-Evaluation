# YOUR CODE HERE
S = input()

# Count uppercase and lowercase letters
uppercase_count = 0
lowercase_count = 0

for char in S:
    if char.isupper():
        uppercase_count += 1
    else:
        lowercase_count += 1

# Apply conversion based on counts
if uppercase_count > lowercase_count:
    # Convert all lowercase to uppercase
    result = S.upper()
else:
    # Convert all uppercase to lowercase
    result = S.lower()

print(result)