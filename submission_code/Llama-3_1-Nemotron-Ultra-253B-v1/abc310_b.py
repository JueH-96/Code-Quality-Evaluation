n, m = map(int, input().split())
products = []
for _ in range(n):
    parts = list(map(int, input().split()))
    p = parts[0]
    c = parts[1]
    f = set(parts[2:2+c])
    products.append((p, f))

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        pi, fi = products[i]
        pj, fj = products[j]
        if pi >= pj and fi.issubset(fj):
            if pi > pj or len(fj) > len(fi):
                print("Yes")
                exit()

print("No")