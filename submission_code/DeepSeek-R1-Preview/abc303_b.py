n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    line = list(map(int, input().split()))
    for i in range(n - 1):
        a = line[i]
        b = line[i + 1]
        if a < b:
            pair = (a, b)
        else:
            pair = (b, a)
        adjacent_pairs.add(pair)

total_pairs = n * (n - 1) // 2
answer = total_pairs - len(adjacent_pairs)
print(answer)