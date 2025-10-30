import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N < 5:
        print(-1)
        return
    
    adj = [[] for _ in range(N + 1)]
    edges = []
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
        edges.append((a, b))
    
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(adj[u])
    
    # Check if there's at least one node with degree >=4
    has_s_node = any(degree[u] >= 4 for u in range(1, N + 1))
    if not has_s_node:
        print(-1)
        return
    
    max_val = 5  # Case 1 by default
    
    # Check Case 2 (two adjacent nodes of degree >=4)
    for a, b in edges:
        if degree[a] >=4 and degree[b] >=4:
            max_val = 8
            break  # No need to check further
    
    # Check Case 3 (a node with at least two degree>=4 neighbors)
    if max_val < 11:
        for u in range(1, N + 1):
            if degree[u] >=4:
                cnt = 0
                for v in adj[u]:
                    if degree[v] >=4:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt >= 2:
                    max_val = 11
                    break
    
    print(max_val)

if __name__ == '__main__':
    main()