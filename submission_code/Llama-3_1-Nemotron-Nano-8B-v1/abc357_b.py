s = input().strip()
upper = 0
lower = 0
for c in s:
    if c.isupper():
        upper += 1
    else:
        lower += 1
if upper > lower:
    new_s = ''.join([char.upper() if char.islower() else char for char in s])
else:
    new_s = ''.join([char.lower() if char.isupper() else char for char in s])
print(new_s)