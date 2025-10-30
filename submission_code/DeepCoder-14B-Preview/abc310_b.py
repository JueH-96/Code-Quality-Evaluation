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
        p_i, f_i = products[i]
        p_j, f_j = products[j]
        if p_i >= p_j:
            if f_i.issubset(f_j):
                if (p_i > p_j) or (len(f_j) > len(f_i)):
                    found = True
                    print("Yes")
                    exit()

print("No")