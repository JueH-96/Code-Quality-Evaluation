n = int(input())
a = list(map(int, input().split()))
total_sum = sum(a)
k = total_sum // n
r = total_sum % n
a_sorted = sorted(a)
res = 0
for i in range(n):
    if i < (n - r):
        target = k
    else:
        target = k + 1
    if a_sorted[i] > target:
        res += a_sorted[i] - target
print(res)