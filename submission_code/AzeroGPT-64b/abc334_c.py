n, k = map(int, input().split())
a = list(map(int, input().split()))

if k % 2:
    res = (a[-1] - a[0])
    c = k + 1
else:
    res = (a[-1] - a[0]) - 1
    c = k

for i in range(1, (n - c) // 2 + 1):
    res += min(a[i - 1] + a[-i - 1], a[-i] - a[i])

print(res)