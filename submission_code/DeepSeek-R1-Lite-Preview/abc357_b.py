S = input().strip()

uppercase_count = sum(1 for c in S if c.isupper())
lowercase_count = sum(1 for c in S if c.islower())

if uppercase_count > lowercase_count:
    # Convert lowercase to uppercase
    new_S = ''.join([c.upper() if c.islower() else c for c in S])
else:
    # Convert uppercase to lowercase
    new_S = ''.join([c.lower() if c.isupper() else c for c in S])

print(new_S)