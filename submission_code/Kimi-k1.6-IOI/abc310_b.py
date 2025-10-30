n, m = map(int, input().split())
products = []
for _ in range(n):
    parts = list(map(int, input().split()))
    p = parts[0]
    c = parts[1]
    fs = set(parts[2:2+c])
    products.append((p, fs))

found = False
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        pi, fi = products[i]
        pj, fj = products[j]
        if pi >= pj and fi.issubset(fj):
            if pi > pj or fi != fj:
                found = True
                break
    if found:
        break

print("Yes" if found else "No")