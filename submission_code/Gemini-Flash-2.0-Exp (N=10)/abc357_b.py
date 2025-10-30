s = input()
upper_count = 0
lower_count = 0
for char in s:
    if char.isupper():
        upper_count += 1
    elif char.islower():
        lower_count += 1

if upper_count > lower_count:
    print(s.upper())
else:
    print(s.lower())