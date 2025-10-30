def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    edges = data[2:]
    
    # Build reachability matrix
    # reachable[i][j] = True means "i is known to be stronger than j"
    reachable = [[False]*N for _ in range(N)]
    
    # Read given edges
    idx = 0
    for _ in range(M):
        a = int(edges[idx]) - 1
        b = int(edges[idx+1]) - 1
        idx += 2
        reachable[a][b] = True
    
    # Floyd-Warshall to compute transitive closure
    for k in range(N):
        row_k = reachable[k]
        for i in range(N):
            if reachable[i][k]:
                # If i > k and k > j then i > j
                row_i = reachable[i]
                for j in range(N):
                    if row_k[j]:
                        row_i[j] = True
    
    # A node i can be the strongest iff there is NO j != i such that j > i is known.
    candidates = []
    for i in range(N):
        # check if any j != i has reachable[j][i] == True
        has_pred = False
        for j in range(N):
            if j != i and reachable[j][i]:
                has_pred = True
                break
        if not has_pred:
            candidates.append(i)
    
    if len(candidates) == 1:
        # Output 1-based index of the unique candidate
        print(candidates[0] + 1)
    else:
        print(-1)


if __name__ == "__main__":
    main()