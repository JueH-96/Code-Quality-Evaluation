def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    idx = 2

    products = []
    for _ in range(N):
        P = int(input_data[idx]); idx += 1
        C = int(input_data[idx]); idx += 1
        funcs = set(map(int, input_data[idx:idx + C]))
        idx += C
        products.append((P, funcs))

    for i in range(N):
        Pi, Fi = products[i]
        for j in range(N):
            if i == j:
                continue
            Pj, Fj = products[j]
            # Conditions:
            # 1) P_i >= P_j
            # 2) F_i is subset of F_j
            # 3) (P_i > P_j) or (F_j contains some function not in F_i)
            if Pi >= Pj and Fi.issubset(Fj) and (Pi > Pj or (len(Fj) > len(Fi))):
                print("Yes")
                return
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()