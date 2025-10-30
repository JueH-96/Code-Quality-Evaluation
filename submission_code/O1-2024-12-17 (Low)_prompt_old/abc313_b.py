def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    edges = data[2:]
    
    # dist[i][j] = True means "i is stronger than j" (i->j)
    dist = [[False]*N for _ in range(N)]
    
    idx = 0
    for _ in range(M):
        A = int(edges[idx]) - 1
        B = int(edges[idx+1]) - 1
        idx += 2
        dist[A][B] = True
    
    # Compute transitive closure: Floyd-Warshall-like
    for k in range(N):
        for i in range(N):
            if dist[i][k]:
                for j in range(N):
                    if dist[k][j]:
                        dist[i][j] = True
    
    # A potential top candidate is one that no one beats.
    candidates = []
    for i in range(N):
        # Check if there's any j such that j beats i
        beaten = False
        for j in range(N):
            if j != i and dist[j][i]:
                beaten = True
                break
        if not beaten:
            candidates.append(i+1)  # store 1-based
    
    # If exactly one candidate, print it; otherwise -1
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

def main():
    solve()

if __name__ == "__main__":
    main()