# YOUR CODE HERE
S = input()

uppercase_count = sum(1 for c in S if c.isupper())
lowercase_count = sum(1 for c in S if c.islower())

if uppercase_count > lowercase_count:
    print(S.upper())
else:
    print(S.lower())