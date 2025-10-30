def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    index = 2

    products = []
    for _ in range(N):
        P = int(data[index]); index += 1
        C = int(data[index]); index += 1
        funcs = list(map(int, data[index:index + C]))
        index += C
        products.append((P, set(funcs)))

    # Check for a pair (i, j) such that:
    # 1) P_i >= P_j
    # 2) F_i subset of F_j
    # 3) (P_i > P_j) or (F_j has at least one function not in F_i)
    # If such a pair exists, print Yes, else No

    for i in range(N):
        Pi, Fi = products[i]
        for j in range(N):
            if i == j:
                continue
            Pj, Fj = products[j]
            if Pi >= Pj:
                if Fi.issubset(Fj):
                    if Pi > Pj or not Fj.issubset(Fi):
                        print("Yes")
                        return

    print("No")

if __name__ == "__main__":
    main()