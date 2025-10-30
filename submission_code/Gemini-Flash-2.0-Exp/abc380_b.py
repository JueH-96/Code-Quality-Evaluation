s = input()
a = []
count = 0
for i in range(1, len(s)):
    if s[i] == '-':
        count += 1
    else:
        a.append(count)
        count = 0
print(*a)