# YOUR CODE HERE
s = input()
upper_count = sum(1 for c in s if c.isupper())
lower_count = len(s) - upper_count

if upper_count > lower_count:
    print(s.lower())
else:
    print(s.upper())