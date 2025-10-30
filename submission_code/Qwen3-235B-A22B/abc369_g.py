import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    from collections import defaultdict

    tree = defaultdict(list)
    for _ in range(N - 1):
        u, v, l = map(int, sys.stdin.readline().split())
        tree[u].append((v, l))
        tree[v].append((u, l))

    # Build rooted tree at 1, compute distance for each node
    parent = [0] * (N + 1)
    distance = [0] * (N + 1)
    visited = [False] * (N + 1)

    stack = [(1, 0, 0)]  # node, parent, cumulative distance
    visited[1] = True

    while stack:
        u, p, d = stack.pop()
        parent[u] = p
        distance[u] = d
        for v, l in tree[u]:
            if not visited[v] and v != p:
                visited[v] = True
                stack.append((v, u, d + l))

    # Collect all nodes except root and sort by distance descending
    nodes = []
    for i in range(2, N + 1):
        nodes.append(distance[i])
    nodes.sort(reverse=True)

    # Precompute prefix sums
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        if i <= len(nodes):
            prefix[i] = prefix[i-1] + nodes[i-1]
        else:
            prefix[i] = prefix[i-1]

    # For each K, answer is 2 * prefix[K]
    for K in range(1, N + 1):
        print(prefix[K] * 2)

threading.Thread(target=main).start()