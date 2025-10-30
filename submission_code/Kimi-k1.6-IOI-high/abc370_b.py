n = int(input())
a = []
for _ in range(n):
    row = list(map(int, input().split()))
    a.append(row)
current = 1
for k in range(1, n+1):
    if current >= k:
        current = a[current-1][k-1]
    else:
        current = a[k-1][current-1]
print(current)