s = input()
upper_count = 0
lower_count = 0
for char in s:
    if 'A' <= char <= 'Z':
        upper_count += 1
    else:
        lower_count += 1

if upper_count > lower_count:
    result = ""
    for char in s:
        if 'a' <= char <= 'z':
            result += chr(ord(char) - 32)
        else:
            result += char
    print(result)
else:
    result = ""
    for char in s:
        if 'A' <= char <= 'Z':
            result += chr(ord(char) + 32)
        else:
            result += char
    print(result)