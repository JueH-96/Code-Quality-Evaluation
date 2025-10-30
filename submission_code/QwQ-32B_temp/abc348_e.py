import sys
from sys import stdin
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    C = list(map(int, stdin.readline().split()))
    
    root = 1
    # Build children and parent arrays via BFS
    children = [[] for _ in range(N+1)]
    parent = [0]*(N+1)
    visited = [False]*(N+1)
    q = deque()
    q.append(root)
    visited[root] = True
    parent[root] = -1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                children[u].append(v)
                visited[v] = True
                q.append(v)
    
    # Compute subtree_C using iterative post_order
    subtree_C = [0]*(N+1)
    stack = []
    stack.append( (root, False) )
    while stack:
        node, visited_flag = stack.pop()
        if not visited_flag:
            stack.append( (node, True) )
            for child in children[node]:
                stack.append( (child, False) )
        else:
            subtree_C[node] = C[node-1]
            for child in children[node]:
                subtree_C[node] += subtree_C[child]
    
    # Compute f_root via BFS
    f_root = 0
    q = deque()
    q.append( (root, 0) )
    while q:
        u, d = q.popleft()
        f_root += C[u-1] * d
        for v in children[u]:
            q.append( (v, d+1) )
    
    total_C = sum(C)
    # Now compute all f using BFS
    f = [0]*(N+1)
    f[root] = f_root
    min_f = f_root
    visited_bfs = [False]*(N+1)
    parent_bfs = [0]*(N+1)
    q = deque()
    q.append(root)
    visited_bfs[root] = True
    parent_bfs[root] = -1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != parent_bfs[u]:
                if not visited_bfs[v]:
                    f_v = f[u] + (total_C - 2 * subtree_C[v])
                    f[v] = f_v
                    if f_v < min_f:
                        min_f = f_v
                    parent_bfs[v] = u
                    visited_bfs[v] = True
                    q.append(v)
    print(min_f)

if __name__ == "__main__":
    main()