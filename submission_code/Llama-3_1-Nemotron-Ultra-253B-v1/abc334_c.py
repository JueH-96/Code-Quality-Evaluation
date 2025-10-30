n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
total = 0
for i in range(1, k, 2):
    total += a[i] - a[i-1]
print(total)