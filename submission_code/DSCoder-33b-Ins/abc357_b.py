# YOUR CODE HERE

S = input()
upper_count = sum(1 for c in S if c.isupper())
lower_count = sum(1 for c in S if c.islower())

if upper_count > lower_count:
    print(S.lower())
else:
    print(S.upper())