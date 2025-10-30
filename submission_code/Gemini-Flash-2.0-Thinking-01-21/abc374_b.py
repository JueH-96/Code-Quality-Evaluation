s = input()
t = input()

n = len(s)
m = len(t)
min_len = min(n, m)

for i in range(min_len):
    if s[i] != t[i]:
        print(i + 1)
        exit()

if n != m:
    print(min_len + 1)
else:
    print(0)