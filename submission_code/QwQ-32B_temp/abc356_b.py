n, m = map(int, input().split())
A = list(map(int, input().split()))
totals = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        totals[j] += x[j]

if all(t >= a for t, a in zip(totals, A)):
    print("Yes")
else:
    print("No")