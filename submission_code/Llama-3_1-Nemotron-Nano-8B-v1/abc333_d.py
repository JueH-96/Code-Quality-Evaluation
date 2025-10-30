import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # Build parent and children structure using BFS
    parent = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    q = deque()
    q.append(1)
    parent[1] = -1  # mark root
    while q:
        u = q.popleft()
        for v in edges[u]:
            if parent[v] == 0:
                parent[v] = u
                children[u].append(v)
                q.append(v)
    
    # Compute subtree sizes using iterative post-order traversal
    subtree_sizes = [0] * (N + 1)
    stack = [(1, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            size = 1
            for child in children[node]:
                size += subtree_sizes[child]
            subtree_sizes[node] = size
        else:
            stack.append((node, True))
            # Push children in reverse order to process them in correct order
            for child in reversed(children[node]):
                stack.append((child, False))
    
    # Collect sizes of children of node 1
    child_nodes = children[1]
    if not child_nodes:
        print(1)
        return
    
    sizes = [subtree_sizes[child] for child in child_nodes]
    max_size = max(sizes)
    sum_rest = sum(sizes) - max_size
    print(sum_rest + 1)

if __name__ == "__main__":
    main()