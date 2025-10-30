def solve():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        line = list(map(int, input().split()))
        p = line[0]
        c = line[1]
        f = set(line[2:])
        products.append((p, f))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            p_i, f_i = products[i]
            p_j, f_j = products[j]

            if p_i >= p_j and f_i.issubset(f_j):
                if p_i > p_j or f_i != f_j:
                    print("Yes")
                    return

    print("No")

solve()