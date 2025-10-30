n = int(input())
a = list(map(int, input().split()))

ans = 0
for i in range(n):
    if a[i] < 0:
        ans -= a[i]
        a[i] = 0
    if i > 0 and a[i] > 0 and a[i-1] < 0:
        if a[i] > ans:
            ans = a[i]
    a[i] += ans
print(ans)