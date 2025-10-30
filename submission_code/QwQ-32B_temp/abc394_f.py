import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    
    # Build children list using BFS
    root = 1
    children = [[] for _ in range(N + 1)]
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append(root)
    visited[root] = True
    while q:
        u = q.popleft()
        for v in edges[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                children[u].append(v)
                visited[v] = True
                q.append(v)
    
    # Post-order traversal to compute dp_root and dp_child
    stack = []
    post_order = []
    stack.append((root, False))
    while stack:
        node, processed = stack.pop()
        if not processed:
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))
        else:
            post_order.append(node)
    
    INF = float('-inf')
    dp_root = [INF] * (N + 1)
    dp_child = [INF] * (N + 1)
    
    for u in post_order:
        # Compute dp_child[u]
        lst_child = []
        for v in children[u]:
            lst_child.append(max(1, dp_child[v]))
        if len(lst_child) >= 3:
            lst_child.sort(reverse=True)
            sum3 = sum(lst_child[:3])
            dp_child[u] = 1 + sum3
        else:
            dp_child[u] = INF
        
        # Compute dp_root[u]
        lst_root = []
        for v in children[u]:
            lst_root.append(max(1, dp_child[v]))
        if len(lst_root) >= 4:
            lst_root.sort(reverse=True)
            sum4 = sum(lst_root[:4])
            dp_root[u] = 1 + sum4
        else:
            dp_root[u] = INF
    
    max_root = max(dp_root[1:N+1])
    print(max_root if max_root >= 5 else -1)

if __name__ == '__main__':
    main()