n = int(input())
a = list(map(int, input().split()))
a.sort()
total = sum(a)
k = total // n
r = total % n

result = 0
# Process first (n - r) elements, target is k
for i in range(n - r):
    if a[i] > k:
        result += a[i] - k
# Process last r elements, target is k + 1
for i in range(n - r, n):
    if a[i] > k + 1:
        result += a[i] - (k + 1)

print(result)