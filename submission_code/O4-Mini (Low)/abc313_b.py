def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    # Initialize reachability matrix
    reachable = [[False]*N for _ in range(N)]
    
    # Read known superiorities
    for _ in range(M):
        A, B = map(int, input().split())
        # convert to 0-based
        reachable[A-1][B-1] = True
    
    # Floyd–Warshall to compute transitive closure
    for k in range(N):
        for i in range(N):
            if reachable[i][k]:
                row_i = reachable[i]
                row_k = reachable[k]
                # if i -> k and k -> j then i -> j
                for j in range(N):
                    if row_k[j]:
                        row_i[j] = True
    
    # Count how many candidates can reach all others
    best = -1
    count = 0
    for i in range(N):
        # check if i is known to be stronger than every other
        ok = True
        for j in range(N):
            if i == j:
                continue
            if not reachable[i][j]:
                ok = False
                break
        if ok:
            best = i + 1  # back to 1-based
            count += 1
    
    # If exactly one, print it; otherwise -1
    print(best if count == 1 else -1)


if __name__ == "__main__":
    main()