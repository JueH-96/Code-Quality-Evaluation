S = input()
upper_count = sum(1 for c in S if c.isupper())
lower_count = sum(1 for c in S if c.islower())

if upper_count > lower_count:
    print(S.upper())
else:
    print(S.lower())