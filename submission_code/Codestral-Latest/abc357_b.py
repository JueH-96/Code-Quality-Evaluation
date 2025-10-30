# YOUR CODE HERE
S = input().strip()

upper_count = sum(1 for c in S if c.isupper())
lower_count = sum(1 for c in S if c.islower())

if upper_count > lower_count:
    result = S.lower()
else:
    result = S.upper()

print(result)