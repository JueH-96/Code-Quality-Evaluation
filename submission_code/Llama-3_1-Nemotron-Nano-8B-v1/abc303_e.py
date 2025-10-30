import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # Count degrees
    degree = [0] * (N+1)
    for u in range(1, N+1):
        degree[u] = len(edges[u])
    
    # Count leaves (degree 1)
    S = sum(1 for u in range(1, N+1) if degree[u] == 1)
    
    # Compute M
    M = (N - S + 2) // 3
    
    sum_k = N - M
    
    # Compute the list of k_i's
    res = [2] * (M-1)
    res.append(sum_k - 2*(M-1))
    res.sort()
    
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()