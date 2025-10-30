s = input()
uppercase_count = 0
lowercase_count = 0
for char in s:
    if 'A' <= char <= 'Z':
        uppercase_count += 1
    elif 'a' <= char <= 'z':
        lowercase_count += 1

if uppercase_count > lowercase_count:
    result = s.upper()
else:
    result = s.lower()

print(result)