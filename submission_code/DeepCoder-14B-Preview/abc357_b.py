S = input().strip()

upper_count = 0
lower_count = 0

for c in S:
    if c.isupper():
        upper_count += 1
    elif c.islower():
        lower_count += 1

if upper_count > lower_count:
    new_S = ''.join([c.upper() if c.islower() else c for c in S])
else:
    new_S = ''.join([c.lower() if c.isupper() else c for c in S])

print(new_S)