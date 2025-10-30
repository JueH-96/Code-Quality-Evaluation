n = int(input())
a = list(map(int, input().split()))
total = sum(a)
avg = total // n
r = total % n
a.sort()
ans = 0
for i in range(n - r):
    if a[i] > avg:
        ans += a[i] - avg
for i in range(n - r, n):
    target = avg + 1
    if a[i] > target:
        ans += a[i] - target
print(ans)