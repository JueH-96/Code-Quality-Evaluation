def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    reach = [[False]*(N+1) for _ in range(N+1)]
    
    for _ in range(M):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        reach[A][B] = True
    
    # Floyd-Warshall algorithm to compute reachability
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if reach[i][k] and reach[k][j]:
                    if not reach[i][j]:
                        reach[i][j] = True
    
    candidates = []
    for x in range(1, N+1):
        valid = True
        for y in range(1, N+1):
            if x == y:
                continue
            if reach[y][x]:
                valid = False
                break
        if valid:
            candidates.append(x)
    
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()