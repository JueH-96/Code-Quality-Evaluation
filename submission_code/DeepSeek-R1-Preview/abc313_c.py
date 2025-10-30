n = int(input())
a = list(map(int, input().split()))
s = sum(a)
k = s // n
m = s % n
a.sort()
sum_ops = 0
for i in range(n):
    if i < (n - m):
        target = k
    else:
        target = k + 1
    sum_ops += abs(a[i] - target)
print(sum_ops // 2)