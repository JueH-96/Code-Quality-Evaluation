import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        A, B = map(int, sys.stdin.readline().split())
        adj[A].append(B)
        adj[B].append(A)
    C = list(map(int, sys.stdin.readline().split()))
    C = [0] + C  # C[1..N]

    total_C = sum(C[1:N+1])

    # Build parent, depth, children using BFS
    root = 1
    parent = [0] * (N + 1)
    depth = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    dq = deque([root])
    parent[root] = 0  # root's parent is 0
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v != parent[u]:
                if parent[v] == 0:
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    children[u].append(v)
                    dq.append(v)

    # Compute sum_subtree using iterative post-order DFS
    sum_subtree = [0] * (N + 1)
    stack = [(root, False)]
    while stack:
        u, visited = stack.pop()
        if not visited:
            stack.append((u, True))
            # Push children in reverse order to process left to right
            for v in reversed(children[u]):
                stack.append((v, False))
        else:
            sum_subtree[u] = C[u]
            for v in children[u]:
                sum_subtree[u] += sum_subtree[v]

    # Compute initial sum_f
    sum_f = 0
    for i in range(1, N + 1):
        sum_f += C[i] * depth[i]

    # Compute f_val using BFS
    f_val = [0] * (N + 1)
    f_val[root] = sum_f
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in children[u]:
            f_val[v] = f_val[u] + (total_C - 2 * sum_subtree[v])
            dq.append(v)

    # Find the minimum f_val
    print(min(f_val[1:N+1]))

if __name__ == "__main__":
    main()