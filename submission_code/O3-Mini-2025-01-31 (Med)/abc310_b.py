def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    products = []
    
    for _ in range(N):
        price = int(next(it))
        C = int(next(it))
        funcs = set()
        for _ in range(C):
            funcs.add(int(next(it)))
        products.append((price, funcs))
    
    # For each pair (i, j) check if product j is "strictly superior" to product i.
    # That is, it satisfies:
    # 1. P_i >= P_j.
    # 2. Product j has all functions that product i has (func_i ⊆ func_j).
    # 3. Either: P_i > P_j or product j has an extra function (i.e., len(func_j) > len(func_i)).
    for i in range(N):
        price_i, funcs_i = products[i]
        for j in range(N):
            if i == j:
                continue
            price_j, funcs_j = products[j]
            if funcs_i.issubset(funcs_j):
                if price_i >= price_j:
                    if price_i > price_j or len(funcs_j) > len(funcs_i):
                        print("Yes")
                        return
    print("No")

if __name__ == '__main__':
    main()