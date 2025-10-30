n, m = map(int, input().split())
products = []
for _ in range(n):
    parts = list(map(int, input().split()))
    p = parts[0]
    c = parts[1]
    functions = set(parts[2:2+c])
    products.append((p, functions))

found = False
for i in range(n):
    for j in range(n):
        pi, set_i = products[i]
        pj, set_j = products[j]
        if pi < pj:
            continue
        if not set_i.issubset(set_j):
            continue
        if pi > pj or len(set_j) > len(set_i):
            print("Yes")
            exit()

print("No")