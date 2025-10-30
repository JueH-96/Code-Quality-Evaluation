S = input()

upper_count = 0
lower_count = 0

for char in S:
    if char.isupper():
        upper_count += 1
    elif char.islower():
        lower_count += 1

if upper_count > lower_count:
    # Convert all lowercase to uppercase
    result = ''
    for char in S:
        if char.islower():
            result += char.upper()
        else:
            result += char
else:
    # Convert all uppercase to lowercase
    result = ''
    for char in S:
        if char.isupper():
            result += char.lower()
        else:
            result += char

print(result)