n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    arr = list(map(int, input().split()))
    for a, b in zip(arr, arr[1:]):
        if a > b:
            a, b = b, a
        adjacent_pairs.add((a, b))

total = n * (n - 1) // 2
print(total - len(adjacent_pairs))