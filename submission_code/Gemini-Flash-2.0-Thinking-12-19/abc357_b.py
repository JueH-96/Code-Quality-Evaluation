s = input()
uppercase_count = 0
lowercase_count = 0
for char in s:
    if char.isupper():
        uppercase_count += 1
    elif char.islower():
        lowercase_count += 1

if uppercase_count > lowercase_count:
    print(s.upper())
else:
    print(s.lower())