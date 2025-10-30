s = input().strip()
upper_count = sum(1 for c in s if c.isupper())
if upper_count > len(s) - upper_count:
    print(s.upper())
else:
    print(s.lower())