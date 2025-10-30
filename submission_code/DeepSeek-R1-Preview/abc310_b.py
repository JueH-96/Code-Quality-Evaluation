n, m = map(int, input().split())
products = []
for _ in range(n):
    parts = list(map(int, input().split()))
    p = parts[0]
    c = parts[1]
    funcs = set(parts[2:2+c])
    products.append((p, funcs))

found = False
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        pi, fi = products[i]
        pj, fj = products[j]
        if pi < pj:
            continue
        if not fi.issubset(fj):
            continue
        if pi > pj or len(fj) > len(fi):
            print("Yes")
            exit()

print("No")