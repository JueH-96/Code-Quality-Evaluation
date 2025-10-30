n, m = map(int, input().split())
adjacent = set()

for _ in range(m):
    photo = list(map(int, input().split()))
    for j in range(len(photo) - 1):
        a = photo[j]
        b = photo[j + 1]
        if a < b:
            pair = (a, b)
        else:
            pair = (b, a)
        adjacent.add(pair)

total = n * (n - 1) // 2
print(total - len(adjacent))