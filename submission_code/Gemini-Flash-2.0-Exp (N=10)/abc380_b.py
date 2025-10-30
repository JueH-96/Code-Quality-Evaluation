s = input()
a = []
count = 0
for char in s[1:]:
    if char == '-':
        count += 1
    elif char == '|':
        a.append(count)
        count = 0
print(*a)