def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    products = []
    for _ in range(N):
        data = list(map(int, input().split()))
        price = data[0]
        c = data[1]
        funcs = set(data[2:])
        products.append((price, funcs))

    # We look for a pair (i, j) such that:
    # 1) price_i >= price_j
    # 2) funcs_i is a subset of funcs_j
    # 3) either price_i > price_j OR funcs_j has strictly more functions than funcs_i
    for i in range(N):
        pi, fi = products[i]
        for j in range(N):
            if i == j:
                continue
            pj, fj = products[j]
            if pi >= pj and fi.issubset(fj):
                if pi > pj or len(fj) > len(fi):
                    print("Yes")
                    return

    print("No")


if __name__ == "__main__":
    main()