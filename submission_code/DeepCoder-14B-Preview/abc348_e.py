import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0

    N = int(data[idx])
    idx += 1

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        A = int(data[idx])
        B = int(data[idx + 1])
        adj[A].append(B)
        adj[B].append(A)
        idx += 2

    # Read C values
    C = list(map(int, data[idx:idx + N]))
    # Make C 1-based
    C = [0] + C  # Now C[1] is the first element

    # Compute depth for each node using BFS
    depth = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                depth[v] = depth[u] + 1
                q.append(v)

    initial_sum = sum(C[i] * depth[i] for i in range(1, N + 1))

    # Compute sum_C_subtree using iterative post-order traversal
    sum_C_subtree = [0] * (N + 1)
    stack = [(1, None, False)]
    while stack:
        u, parent, processed = stack.pop()
        if not processed:
            stack.append((u, parent, True))
            for v in adj[u]:
                if v != parent:
                    stack.append((v, u, False))
        else:
            sum_C_subtree[u] = C[u]
            for v in adj[u]:
                if v != parent:
                    sum_C_subtree[u] += sum_C_subtree[v]

    total_C = sum_C_subtree[1]

    # Compute f for all nodes
    f = [0] * (N + 1)
    f[1] = initial_sum

    # BFS to calculate f for all nodes
    queue = deque()
    queue.append((1, -1))  # (current node, parent)
    while queue:
        u, parent_u = queue.popleft()
        for v in adj[u]:
            if v == parent_u:
                continue
            f[v] = f[u] + (total_C - 2 * sum_C_subtree[v])
            queue.append((v, u))

    # Find the minimum f value
    min_f = min(f[1:N + 1])
    print(min_f)

if __name__ == '__main__':
    main()