s = input()
a = []
count = 0
for i in range(1, len(s)):
    if s[i] == '-':
        count += 1
    elif s[i] == '|':
        a.append(count)
        count = 0
print(*a)