import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N == 1:
        print(0)
        return
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Build the tree structure with parent pointers and children
    tree = [[] for _ in range(N + 1)]
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                tree[u].append(v)
                q.append(v)
                visited[v] = True
    
    # Compute the size of each subtree using iterative post-order traversal
    size = [1] * (N + 1)
    stack = []
    visited = [False] * (N + 1)
    stack.append((1, False))
    while stack:
        node, processed = stack.pop()
        if not processed:
            stack.append((node, True))
            # Push children in reverse order to process them in correct order
            for child in reversed(tree[node]):
                stack.append((child, False))
        else:
            for child in tree[node]:
                size[node] += size[child]
    
    # Get the children of the root (1)
    children = tree[1]
    if not children:
        # Root has no children, which is only possible if N=1, but N>=2
        print(1)
        return
    max_size = max(size[child] for child in children)
    print(N - max_size)

if __name__ == '__main__':
    main()