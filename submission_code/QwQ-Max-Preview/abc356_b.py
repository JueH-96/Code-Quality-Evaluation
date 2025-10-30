n, m = map(int, input().split())
a = list(map(int, input().split()))
sums = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        sums[j] += x[j]

all_met = True
for j in range(m):
    if sums[j] < a[j]:
        all_met = False
        break

print("Yes" if all_met else "No")