n, m = map(int, input().split())
a = list(map(int, input().split()))
totals = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        totals[j] += x[j]

result = all(t >= req for t, req in zip(totals, a))
print("Yes" if result else "No")