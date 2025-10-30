from collections import deque
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(N - 1):
        a = int(data[idx])
        b = int(data[idx + 1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    
    C = list(map(int, data[idx:idx + N]))
    idx += N
    
    # Compute depth and f_root using BFS starting from root (node 1)
    root = 1
    depth = [0] * (N + 1)
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    f_root = 0
    q = deque([root])
    visited[root] = True
    depth[root] = 0
    f_root += C[root - 1] * depth[root]
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                f_root += C[v - 1] * depth[v]
                q.append(v)
    
    # Compute sum_subtree using iterative post-order DFS
    sum_subtree = [0] * (N + 1)
    stack = [(root, False)]
    while stack:
        node, is_processed = stack.pop()
        if not is_processed:
            stack.append((node, True))
            # Push children (neighbors except parent) in reverse order to maintain order
            for neighbor in reversed(adj[node]):
                if neighbor != parent[node]:
                    stack.append((neighbor, False))
        else:
            s = C[node - 1]
            for neighbor in adj[node]:
                if neighbor != parent[node]:
                    s += sum_subtree[neighbor]
            sum_subtree[node] = s
    
    total_C = sum(C)
    
    # Compute f values using BFS
    f = [0] * (N + 1)
    f[root] = f_root
    par_f = [0] * (N + 1)
    par_f[root] = 0  # Root has no parent in this BFS
    q = deque([root])
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != par_f[u]:
                f[v] = f[u] + (total_C - 2 * sum_subtree[v])
                par_f[v] = u
                q.append(v)
    
    print(min(f[1:N+1]))

if __name__ == '__main__':
    main()