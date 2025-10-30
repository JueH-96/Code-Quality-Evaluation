S = input().strip()
upper = sum(1 for c in S if c.isupper())
lower = len(S) - upper
if upper > lower:
    print(S.upper())
else:
    print(S.lower())