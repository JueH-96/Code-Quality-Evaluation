s = input()
result = []
count = 0
for char in s:
    if char == '-':
        count += 1
    elif char == '|':
        if count > 0:
            result.append(count)
            count = 0
print(*result)