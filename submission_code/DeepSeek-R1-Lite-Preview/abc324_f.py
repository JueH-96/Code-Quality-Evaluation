import sys
from collections import deque

def main():
    import sys
    import math

    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N+1)]
    in_degree = [0] * (N+1)
    max_ratio = 0.0

    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        adj[u].append((v, b, c))
        in_degree[v] += 1
        ratio = b / c
        if ratio > max_ratio:
            max_ratio = ratio

    # Topological sort using Kahn's algorithm
    queue = deque()
    for i in range(1, N+1):
        if in_degree[i] == 0:
            queue.append(i)
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v, _, _ in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Binary search for the maximum ratio
    left = 0.0
    right = max_ratio
    eps = 1e-10
    for _ in range(100):
        mid = (left + right) / 2
        # Feasibility check
        dist = [ -math.inf ] * (N+1)
        dist[1] = 0.0
        for u in topo_order:
            if dist[u] == -math.inf:
                continue
            for v, b, c in adj[u]:
                if dist[v] < dist[u] + (b - mid * c):
                    dist[v] = dist[u] + (b - mid * c)
        if dist[N] >= 0:
            left = mid
        else:
            right = mid
        if right - left < eps:
            break

    # Print the result with sufficient precision
    print("{0:.15f}".format(left))

if __name__ == '__main__':
    main()