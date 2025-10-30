# YOUR CODE HERE
def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    adj = [[False]*N for _ in range(N)]
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        adj[A-1][B-1] = True
    # Floyd-Warshall for transitive closure
    for k in range(N):
        for i in range(N):
            if adj[i][k]:
                for j in range(N):
                    if adj[k][j]:
                        if not adj[i][j]:
                            adj[i][j] = True
    candidates = []
    for j in range(N):
        has_incoming = False
        for i in range(N):
            if i != j and adj[i][j]:
                has_incoming = True
                break
        if not has_incoming:
            candidates.append(j+1)
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()