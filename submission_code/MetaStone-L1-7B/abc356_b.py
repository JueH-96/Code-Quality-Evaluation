n, m = map(int, input().split())
a = list(map(int, input().split()))
total = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        total[j] += x[j]

all_met = True
for j in range(m):
    if total[j] < a[j]:
        all_met = False
        break

print("Yes" if all_met else "No")