import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N == 1:
        print(1)
        return

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # BFS to assign parents and build children
    parents = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    queue = deque([1])
    parents[1] = -1  # mark root

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if parents[v] == 0 and v != parents[u]:
                parents[v] = u
                children[u].append(v)
                queue.append(v)

    # Compute depth for each node using post-order traversal
    depth = [0] * (N + 1)
    stack = []
    visited = [False] * (N + 1)
    stack.append((1, False))

    while stack:
        node, processed = stack.pop()
        if processed:
            max_child_depth = 0
            for child in children[node]:
                if depth[child] > max_child_depth:
                    max_child_depth = depth[child]
            depth[node] = 1 + max_child_depth
        else:
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))

    # Collect depths of children of 1
    sum_depth = 0
    for u in children[1]:
        sum_depth += depth[u]

    if sum_depth == 0:
        print(1)
    else:
        print(sum_depth)

if __name__ == "__main__":
    main()