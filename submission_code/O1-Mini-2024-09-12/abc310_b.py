# YOUR CODE HERE
def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    products = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        P = parts[0]
        C = parts[1]
        F = set(parts[2:])
        products.append( (P, F) )
    for i in range(N):
        P_i, F_i = products[i]
        for j in range(N):
            if i == j:
                continue
            P_j, F_j = products[j]
            if P_i >= P_j and F_j.issuperset(F_i):
                if P_i > P_j or F_j != F_i:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()