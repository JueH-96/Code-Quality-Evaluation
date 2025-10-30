import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u = int(data[idx])
        v = int(data[idx + 1])
        adj[u].append(v)
        adj[v].append(u)
        idx += 2
    
    # Build children structure using BFS
    children = [[] for _ in range(N + 1)]
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque([1])
    visited[1] = True
    parent[1] = -1
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v] and v != parent[u]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                q.append(v)
    
    # Compute subtree sizes using iterative post-order traversal
    size = [1] * (N + 1)
    stack = [(1, -1, False)]
    
    while stack:
        u, p, visited_flag = stack.pop()
        if not visited_flag:
            stack.append((u, p, True))
            # Push children in reverse order to process left to right
            for v in reversed(children[u]):
                stack.append((v, u, False))
        else:
            for v in children[u]:
                size[u] += size[v]
    
    # Get children of node 1
    children_of_1 = children[1]
    k = len(children_of_1)
    if k == 0:
        print(1)
        return
    
    sizes = [size[v] for v in children_of_1]
    sizes.sort()
    sum_steps = sum(sizes[:k-1]) if k > 1 else 0
    print(sum_steps + 1)

if __name__ == '__main__':
    main()