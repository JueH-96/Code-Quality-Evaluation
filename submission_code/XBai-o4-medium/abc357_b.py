s = input().strip()
upper_count = 0
lower_count = 0
for c in s:
    if c.isupper():
        upper_count += 1
    else:
        lower_count += 1
if upper_count > lower_count:
    print(s.upper())
else:
    print(s.lower())