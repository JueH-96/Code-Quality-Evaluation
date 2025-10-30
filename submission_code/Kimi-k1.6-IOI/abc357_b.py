s = input().strip()
upper = sum(1 for c in s if c.isupper())
lower = sum(1 for c in s if c.islower())
print(s.upper() if upper > lower else s.lower())