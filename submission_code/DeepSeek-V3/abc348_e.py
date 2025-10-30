import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B = map(int, sys.stdin.readline().split())
        edges[A].append(B)
        edges[B].append(A)
    C = list(map(int, sys.stdin.readline().split()))
    
    # Compute the sum of C_i for all nodes
    total_C = sum(C)
    
    # Compute the sum of C_i * depth_i for all nodes, using node 1 as root
    depth = [0] * (N+1)
    parent = [0] * (N+1)
    q = deque()
    q.append(1)
    parent[1] = -1
    sum_C_depth = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if parent[v] == 0 and v != parent[u]:
                parent[v] = u
                depth[v] = depth[u] + 1
                sum_C_depth += C[v-1] * depth[v]
                q.append(v)
    
    # Compute f(1)
    f1 = sum_C_depth
    
    # Now, we need to compute f(v) for all v using the re-rooting technique
    # Initialize f for all nodes
    f = [0] * (N+1)
    f[1] = f1
    
    # Re-rooting: when moving from u to v, f(v) = f(u) - C[v-1] * depth[v] + (total_C - C[v-1]) * 1
    # Because when moving from u to v, the depth of v decreases by 1, and the depth of all other nodes increases by 1
    # So, f(v) = f(u) - C[v-1] * 1 + (total_C - C[v-1]) * 1
    # Which simplifies to f(v) = f(u) - C[v-1] + (total_C - C[v-1])
    # Or f(v) = f(u) + (total_C - 2 * C[v-1])
    
    q = deque()
    q.append(1)
    while q:
        u = q.popleft()
        for v in edges[u]:
            if parent[v] == u:
                f[v] = f[u] + (total_C - 2 * C[v-1])
                q.append(v)
    
    # Find the minimum f(v)
    min_f = min(f[1:])
    print(min_f)

if __name__ == "__main__":
    main()