n, m = map(int, input().split())
a = list(map(int, input().split()))
foods = [list(map(int, input().split())) for _ in range(n)]

found = False
for j in range(m):
    total = 0
    for i in range(n):
        total += foods[i][j]
    if total < a[j]:
        found = True
        break

print("No" if found else "Yes")