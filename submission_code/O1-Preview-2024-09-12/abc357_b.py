# YOUR CODE HERE
S = input()
upper_count = sum(1 for c in S if c.isupper())
lower_count = sum(1 for c in S if c.islower())
if upper_count > lower_count:
    result = ''.join(c.upper() if c.islower() else c for c in S)
else:
    result = ''.join(c.lower() if c.isupper() else c for c in S)
print(result)