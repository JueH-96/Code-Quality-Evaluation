s = input()
d = {}
for i in s:
    if i in d:
        d[i] += 1
    else:
        d[i] = 1
m = max(d.values())
l = []
for i in d:
    if d[i] == m:
        l.append(i)
print(min(l))