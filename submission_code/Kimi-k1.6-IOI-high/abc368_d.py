import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        adj[a].append(b)
        adj[b].append(a)
    
    V = list(map(int, input[ptr:ptr + K]))
    ptr += K
    is_k = [False] * (N + 1)
    for v in V:
        is_k[v] = True
    
    # Build tree rooted at 1 using BFS
    root = 1
    parent = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    queue = deque([root])
    parent[root] = -1  # Mark root's parent
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if parent[u] != v and parent[v] == 0:
                parent[v] = u
                children[u].append(v)
                queue.append(v)
    
    # Compute subtree_K using iterative post-order traversal
    subtree_K = [0] * (N + 1)
    stack = [(root, False)]
    
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            # Push children in reverse order to process leftmost child first
            for child in reversed(children[node]):
                stack.append((child, False))
        else:
            cnt = 1 if is_k[node] else 0
            for child in children[node]:
                cnt += subtree_K[child]
            subtree_K[node] = cnt
    
    total_K = subtree_K[root]
    count = 0
    
    for node in range(1, N + 1):
        if node == root:
            continue
        if subtree_K[node] > 0 and (total_K - subtree_K[node]) > 0:
            count += 1
    
    print(count + 1)

if __name__ == '__main__':
    main()