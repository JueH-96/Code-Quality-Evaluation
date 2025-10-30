# YOUR CODE HERE
S = input().strip()

# Count uppercase and lowercase letters
uppercase_count = sum(1 for char in S if char.isupper())
lowercase_count = sum(1 for char in S if char.islower())

# Apply conversion rule
if uppercase_count > lowercase_count:
    S = S.upper()
else:
    S = S.lower()

print(S)