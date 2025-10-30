def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    products = []
    index = 2
    for i in range(n):
        p = int(data[index])
        c = int(data[index+1])
        index += 2
        funcs = list(map(int, data[index:index+c]))
        index += c
        products.append((p, set(funcs)))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i, set_i = products[i]
            p_j, set_j = products[j]
            if p_i >= p_j:
                if set_i.issubset(set_j):
                    if p_i > p_j or set_i != set_j:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()