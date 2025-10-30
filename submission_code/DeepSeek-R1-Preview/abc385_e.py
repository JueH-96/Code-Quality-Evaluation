import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N == 1:
        print(0)
        return
    adj = [[] for _ in range(N + 1)]  # 1-based
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute degree
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(adj[u])
    
    # Precompute non_leaf_neighbors for each node
    non_leaf_neighbors = [0] * (N + 1)
    for u in range(1, N + 1):
        cnt = 0
        for v in adj[u]:
            if degree[v] > 1:
                cnt += 1
        non_leaf_neighbors[u] = cnt
    
    min_del = float('inf')
    # Iterate each node as candidate C
    for C in range(1, N + 1):
        # Collect all adjacent B where non_leaf_neighbors[B] == 1
        valid_bs = []
        for B in adj[C]:
            if non_leaf_neighbors[B] == 1:
                valid_bs.append(B)
        if not valid_bs:
            continue
        # Group by y
        freq = defaultdict(int)
        for B in valid_bs:
            y = degree[B] - 1
            freq[y] += 1
        # Check all possible y groups
        for y in freq:
            x = freq[y]
            if x < 1:
                continue
            size = 1 + x * (y + 1)
            deletions = N - size
            if deletions < min_del:
                min_del = deletions
    if min_del == float('inf'):
        # No solution found, but problem says it's always possible
        # So this case shouldn't happen
        print(N - 1)  # minimal case x=1, y=0
    else:
        print(min_del)

if __name__ == '__main__':
    main()