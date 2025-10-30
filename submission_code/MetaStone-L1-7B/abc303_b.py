n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    people = list(map(int, input().split()))
    for i in range(n - 1):
        a = people[i]
        b = people[i + 1]
        if a < b:
            pair = (a, b)
        else:
            pair = (b, a)
        adjacent_pairs.add(pair)

total_pairs = n * (n - 1) // 2
result = total_pairs - len(adjacent_pairs)
print(result)