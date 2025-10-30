n, m = map(int, input().split())
adjacent = set()

for _ in range(m):
    row = list(map(int, input().split()))
    for i in range(n - 1):
        x, y = row[i], row[i+1]
        pair = tuple(sorted((x, y)))
        adjacent.add(pair)

total = n * (n - 1) // 2
print(total - len(adjacent))