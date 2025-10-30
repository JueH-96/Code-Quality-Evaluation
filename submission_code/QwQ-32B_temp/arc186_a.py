def main():
    import sys
    N, Q = map(int, sys.stdin.readline().split())
    K_list = [int(sys.stdin.readline()) for _ in range(Q)]
    
    possible = set()
    
    for a in range(N + 1):
        for c in range(N - a + 1):
            for b in range(N + 1):
                for d in range(N - b + 1):
                    m = N - a - c
                    n = N - b - d
                    if m < 0 or n < 0:
                        continue
                    L = (a + c) * N + (b + d) * N - (a + c) * (b + d)
                    U = L + m * n
                    for k in range(L, U + 1):
                        possible.add(k)
    
    for K in K_list:
        print("Yes" if K in possible else "No")

if __name__ == "__main__":
    main()