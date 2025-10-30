def main():
    import sys
    tokens = sys.stdin.read().split()
    N = int(tokens[0])
    M = int(tokens[1])
    index = 2
    products = []
    for _ in range(N):
        P_i = int(tokens[index])
        C_i = int(tokens[index + 1])
        functions_i = set(map(int, tokens[index + 2 : index + 2 + C_i]))
        products.append((P_i, functions_i))
        index += 2 + C_i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            P_i, functions_i = products[i]
            P_j, functions_j = products[j]
            if P_i >= P_j and functions_j.issuperset(functions_i):
                if P_i > P_j or (functions_j - functions_i):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()