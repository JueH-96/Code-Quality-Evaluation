import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    parent = [0] * (N+1)
    children = defaultdict(list)
    visited = [False] * (N+1)
    stack = [1]
    visited[1] = True
    parent[1] = -1  # Marking root
    
    # Build the tree structure as a parent-children structure
    while stack:
        u = stack.pop()
        for v in edges[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                children[u].append(v)
                visited[v] = True
                stack.append(v)
    
    # Compute leaves_count using post-order traversal
    leaves_count = [0] * (N + 1)
    stack = []
    visited = [False] * (N + 1)
    stack.append((1, False))
    
    while stack:
        node, processed = stack.pop()
        if processed:
            if len(children[node]) == 0:
                leaves_count[node] = 1
            else:
                cnt = 0
                for child in children[node]:
                    cnt += leaves_count[child]
                leaves_count[node] = cnt
        else:
            stack.append((node, True))
            for child in children[node]:
                stack.append((child, False))
    
    max_size = 0
    for u in range(1, N+1):
        if len(children[u]) == 0:
            continue
        freq = defaultdict(int)
        for v in children[u]:
            freq[leaves_count[v]] += 1
        for y in freq:
            x = freq[y]
            size = 1 + x + x * y
            if size > max_size:
                max_size = size
    
    print(N - max_size)

if __name__ == '__main__':
    main()