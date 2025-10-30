s = input().strip()
a = []
prev = -1
for i in range(len(s)):
    if s[i] == '|':
        if prev == -1:
            a.append(i)
        else:
            a.append(i - prev - 1)
        prev = i
print(' '.join(map(str, a)))