s = input().strip()
upper = sum(1 for c in s if c.isupper())
lower = len(s) - upper
if upper > lower:
    print(s.upper())
else:
    print(s.lower())