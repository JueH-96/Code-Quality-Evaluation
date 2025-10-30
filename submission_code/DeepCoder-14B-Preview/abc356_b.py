n, m = map(int, input().split())
a = list(map(int, input().split()))
totals = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        totals[j] += x[j]

met = True
for j in range(m):
    if totals[j] < a[j]:
        met = False
        break

print("Yes" if met else "No")