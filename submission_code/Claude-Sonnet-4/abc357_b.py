# YOUR CODE HERE
S = input().strip()

uppercase_count = sum(1 for c in S if c.isupper())
lowercase_count = sum(1 for c in S if c.islower())

if uppercase_count > lowercase_count:
    # Convert all lowercase to uppercase
    result = S.upper()
else:
    # Convert all uppercase to lowercase
    result = S.lower()

print(result)