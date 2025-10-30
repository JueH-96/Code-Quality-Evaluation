def solve():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        line = list(map(int, input().split()))
        products.append((line[0], set(line[2:])))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            pi, fi = products[i]
            pj, fj = products[j]

            if pi >= pj and fi.issubset(fj):
                if pi > pj or len(fj - fi) > 0:
                    print("Yes")
                    return

    print("No")

solve()