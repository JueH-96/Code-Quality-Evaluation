def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    products = []
    for _ in range(N):
        P_i = int(input[ptr])
        ptr += 1
        C_i = int(input[ptr])
        ptr += 1
        F_list = list(map(int, input[ptr:ptr + C_i]))
        ptr += C_i
        F_set = set(F_list)
        products.append((P_i, F_set))

    found = False
    for i in range(N):
        p_i, f_i = products[i]
        for j in range(N):
            if i == j:
                continue
            p_j, f_j = products[j]
            if p_i >= p_j:
                if f_i.issubset(f_j):
                    if (p_i > p_j) or (len(f_j) > len(f_i)):
                        found = True
                        break
        if found:
            break

    print("Yes" if found else "No")

if __name__ == "__main__":
    main()