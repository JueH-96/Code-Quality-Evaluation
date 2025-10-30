import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    C = list(map(int, sys.stdin.readline().split()))
    total_c = sum(C)
    
    # Compute sum_c_subtree via iterative post-order traversal
    sum_c_subtree = [0] * (N + 1)
    stack = [(1, -1, False)]
    while stack:
        node, parent, is_processed = stack.pop()
        if not is_processed:
            stack.append((node, parent, True))
            for neighbor in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, False))
        else:
            sum_c = C[node - 1]
            for neighbor in adj[node]:
                if neighbor != parent:
                    sum_c += sum_c_subtree[neighbor]
            sum_c_subtree[node] = sum_c
    
    # Compute depth and parent array via BFS
    depth = [0] * (N + 1)
    parent = [-1] * (N + 1)
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                depth[v] = depth[u] + 1
                queue.append(v)
    
    # Compute initial f_root
    f_root = sum(depth[i] * C[i-1] for i in range(1, N+1))
    min_f = f_root
    
    # Compute f for all nodes via BFS
    f = [0] * (N + 1)
    f[1] = f_root
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v != parent[u]:
                f[v] = f[u] + (total_c - 2 * sum_c_subtree[v])
                if f[v] < min_f:
                    min_f = f[v]
                queue.append(v)
    
    print(min_f)

if __name__ == "__main__":
    main()