t = int(input().strip())
results = []

for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().split()))
    best = -1
    for i in range(n):
        prod = 1
        for j in range(n):
            if j == i:
                prod *= (a[j] + 1)
            else:
                prod *= a[j]
        if prod > best:
            best = prod
    results.append(best)

for res in results:
    print(res)