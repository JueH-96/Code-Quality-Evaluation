import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    V = list(map(int, sys.stdin.readline().split()))
    
    # Build parent and children arrays using BFS
    root = 1
    parent = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    visited = [False] * (N + 1)
    q = deque()
    q.append(root)
    visited[root] = True
    parent[root] = 0  # root has no parent
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                children[u].append(v)
                visited[v] = True
                q.append(v)
    
    # Post-order traversal using iterative method
    post_order = []
    stack = []
    stack.append((root, False))
    while stack:
        node, processed = stack.pop()
        if processed:
            post_order.append(node)
        else:
            stack.append((node, True))
            # Push children in reverse to process them in order
            for child in reversed(children[node]):
                stack.append((child, False))
    
    # Compute count array
    is_required = [False] * (N + 1)
    for v in V:
        is_required[v] = True
    count = [0] * (N + 1)
    for node in post_order:
        cnt = is_required[node]
        for child in children[node]:
            cnt += count[child]
        count[node] = cnt
    
    # Calculate edges_count
    edges_count = 0
    for u in range(1, N + 1):
        if u != root:
            if count[u] > 0 and (K - count[u] > 0):
                edges_count += 1
    print(edges_count + 1)

if __name__ == "__main__":
    main()