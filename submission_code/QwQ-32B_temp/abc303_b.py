n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    a = list(map(int, input().split()))
    for i in range(n - 1):
        x = a[i]
        y = a[i + 1]
        if x > y:
            x, y = y, x
        adjacent_pairs.add((x, y))

total_pairs = n * (n - 1) // 2
print(total_pairs - len(adjacent_pairs))