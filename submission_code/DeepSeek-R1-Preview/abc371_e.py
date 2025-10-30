n = int(input())
a = list(map(int, input().split()))
last = {}
res = 0
for i in range(n):
    num = a[i]
    p = last.get(num, -1)
    res += (i - p) * (n - i)
    last[num] = i
print(res)