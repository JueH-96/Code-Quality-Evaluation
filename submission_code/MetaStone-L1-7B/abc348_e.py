import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]  # Using 1-based indexing
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    C = list(map(int, sys.stdin.readline().split()))
    S = sum(C)
    
    # Compute T using iterative post-order traversal
    T = [0] * (N + 1)
    stack = []
    stack.append((1, -1, False))
    while stack:
        node, parent, processed = stack.pop()
        if not processed:
            stack.append((node, parent, True))
            for neighbor in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, False))
        else:
            T[node] = C[node - 1]
            for neighbor in adj[node]:
                if neighbor != parent:
                    T[node] += T[neighbor]
    
    # Compute f_root using BFS
    depth = [0] * (N + 1)
    parent = [0] * (N + 1)
    f_root = 0
    queue = deque()
    queue.append((1, 0))
    parent[1] = -1
    while queue:
        u, d = queue.popleft()
        f_root += C[u - 1] * d
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                depth[v] = d + 1
                queue.append((v, d + 1))
    
    # Compute f for all nodes using BFS
    f = [0] * (N + 1)
    f[1] = f_root
    queue = deque()
    queue.append((1, -1))
    while queue:
        u, p = queue.popleft()
        for v in adj[u]:
            if v != p:
                f[v] = f[u] + (S - 2 * T[v])
                queue.append((v, u))
    
    # Find the minimum f value
    min_f = min(f[1:])
    print(min_f)

if __name__ == "__main__":
    main()