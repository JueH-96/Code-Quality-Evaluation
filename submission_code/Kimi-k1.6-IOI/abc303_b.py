n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    row = list(map(int, input().split()))
    for i in range(n - 1):
        a, b = row[i], row[i + 1]
        if a > b:
            a, b = b, a
        adjacent_pairs.add((a, b))

total_pairs = n * (n - 1) // 2
print(total_pairs - len(adjacent_pairs))