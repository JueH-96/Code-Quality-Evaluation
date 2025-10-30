import sys
import heapq
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))

    # Compute distance from 1 to each node
    distance = [0] * (N+1)
    parent = [0] * (N+1)
    stack = [1]
    visited = [False] * (N+1)
    visited[1] = True
    while stack:
        u = stack.pop()
        for v, l in edges[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                distance[v] = distance[u] + l
                stack.append(v)

    # Euler Tour to get in_time and out_time for subtree management
    in_time = [0] * (N+1)
    out_time = [0] * (N+1)
    time = 0
    stack = [(1, False)]
    while stack:
        u, visited = stack.pop()
        if visited:
            out_time[u] = time
            continue
        time += 1
        in_time[u] = time
        stack.append((u, True))
        for v, _ in reversed(edges[u]):
            if v != parent[u]:
                stack.append((v, False))

    # Segment Tree for range max update and point query
    size = 1
    while size < N+2:
        size <<= 1
    seg = [0] * (2 * size)

    def update(l, r, val):
        l += size
        r += size
        while l < r:
            if l % 2:
                seg[l] = max(seg[l], val)
                l += 1
            if r % 2:
                r -= 1
                seg[r] = max(seg[r], val)
            l >>= 1
            r >>= 1

    def query(pos):
        pos += size
        res = 0
        while pos > 0:
            res = max(res, seg[pos])
            pos >>= 1
        return res

    # Priority queue (max-heap)
    heap = []
    for u in range(1, N+1):
        heapq.heappush(heap, (-distance[u], u))

    sum_contribution = 0
    res = [0] * (N+1)
    selected = 0

    while heap and selected < N:
        current_contrib_neg, u = heapq.heappop(heap)
        current_contrib = -current_contrib_neg
        max_ancestor = query(in_time[u])
        actual_contrib = distance[u] - max_ancestor
        if actual_contrib <= 0:
            continue
        sum_contribution += actual_contrib
        selected += 1
        res[selected] = sum_contribution * 2
        # Update the subtree of u with max_ancestor = distance[u]
        update(in_time[u], out_time[u] + 1, distance[u])

    # Fill the rest with the maximum possible sum
    for k in range(1, N+1):
        if res[k] == 0:
            res[k] = res[selected]

    for k in range(1, N+1):
        print(res[k])

if __name__ == '__main__':
    main()