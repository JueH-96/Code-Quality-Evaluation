s = input()
t = input()

n = len(s)
m = len(t)

for i in range(min(n, m)):
    if s[i] != t[i]:
        print(i + 1)
        exit()

if n != m:
    print(min(n, m) + 1)
else:
    print(0)