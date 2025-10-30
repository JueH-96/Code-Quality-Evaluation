n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    line = list(map(int, input().split()))
    for i in range(n - 1):
        a, b = line[i], line[i+1]
        pair = tuple(sorted((a, b)))
        adjacent_pairs.add(pair)

total_pairs = n * (n - 1) // 2
answer = total_pairs - len(adjacent_pairs)
print(answer)