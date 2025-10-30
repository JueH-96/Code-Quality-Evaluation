def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    products = []
    for _ in range(N):
        parts = list(map(int, input().split()))
        price = parts[0]
        C = parts[1]
        funcs = set(parts[2:])
        products.append((price, funcs))
    
    # Check each pair of products (i, j)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            price_i, funcs_i = products[i]
            price_j, funcs_j = products[j]
            # Condition 1: price[i] >= price[j]
            if price_i < price_j:
                continue
            # Condition 2: product j has all functions of product i (i.e. funcs[i] is subset of funcs[j])
            if not funcs_i.issubset(funcs_j):
                continue
            # Condition 3: Either price_i > price_j or funcs_j has some functions which i's product lacks
            if price_i > price_j or (funcs_j - funcs_i):
                print("Yes")
                return
    print("No")

if __name__ == '__main__':
    main()