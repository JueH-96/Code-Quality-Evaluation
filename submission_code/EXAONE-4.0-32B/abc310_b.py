def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    index = 2
    products = []
    for _ in range(n):
        price = int(data[index])
        index += 1
        c = int(data[index])
        index += 1
        funcs = list(map(int, data[index:index+c]))
        index += c
        products.append((price, set(funcs)))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i, s_i = products[i]
            p_j, s_j = products[j]
            if p_i >= p_j and s_i.issubset(s_j):
                if p_i > p_j or s_i != s_j:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()