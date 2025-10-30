def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    products = []
    for _ in range(N):
        price = int(next(it))
        count = int(next(it))
        funcs = set()
        for _ in range(count):
            funcs.add(int(next(it)))
        products.append((price, funcs))
    
    # We want to check whether there exists a pair (i, j) with i != j such that:
    # - P_i >= P_j, and
    # - The function set of product i is a subset of the function set of product j, and
    # - Either P_i > P_j or product j has at least one function that product i does not (i.e. the function sets differ).
    #
    # In other words, product j is "strictly superior" to product i.
    for i in range(N):
        price_i, funcs_i = products[i]
        for j in range(N):
            if i == j: 
                continue
            price_j, funcs_j = products[j]
            if price_i >= price_j and funcs_i.issubset(funcs_j) and (price_i > price_j or funcs_i != funcs_j):
                print("Yes")
                return
    print("No")
    
if __name__ == '__main__':
    main()