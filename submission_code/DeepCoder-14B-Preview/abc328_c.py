n, q = map(int, input().split())
s = input().strip()

if n <= 1:
    a = []
else:
    a = [0] * (n - 1)
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            a[i] = 1

prefix = [0] * (len(a) + 1)
for i in range(1, len(a) + 1):
    prefix[i] = prefix[i - 1] + a[i - 1]

for _ in range(q):
    l, r = map(int, input().split())
    if l >= r:
        print(0)
    else:
        res = prefix[r - 1] - prefix[l - 1]
        print(res)