n, m = map(int, input().split())
a = list(map(int, input().split()))
foods = [list(map(int, input().split())) for _ in range(n)]

for j in range(m):
    total = sum(row[j] for row in foods)
    if total < a[j]:
        print("No")
        exit()
print("Yes")