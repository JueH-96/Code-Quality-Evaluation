import sys
import heapq

def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(1 << 25)

    N, M = map(int, sys.stdin.readline().split())
    if N == 1:
        print(1)
        return
    p = list(map(int, sys.stdin.readline().split()))
    tree = [[] for _ in range(N + 1)]
    for i in range(1, N - 1):
        tree[p[i]].append(i + 2)
    p = [0] + [p[i-1] for i in range(1, N)] + [0]

    insurances = []
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        insurances.append((x, y))

    # DFS to assign entry and exit times and depths
    entry = [0] * (N + 1)
    exit = [0] * (N + 1)
    depth = [0] * (N + 1)
    time = 0

    def dfs(node, d):
        nonlocal time
        time += 1
        entry[node] = time
        depth[node] = d
        for child in tree[node]:
            dfs(child, d + 1)
        exit[node] = time

    dfs(1, 0)

    # Sort insurances by entry time
    insurances_sorted = []
    for x, y in insurances:
        insurances_sorted.append((entry[x], depth[x] + y))
    insurances_sorted.sort()

    # Traverse DFS order and use priority queue to manage coverage
    covered = [False] * (time + 1)
    idx = 0
    pq = []
    dfs_order = []

    def euler_tour(node):
        nonlocal dfs_order
        dfs_order.append(node)
        for child in tree[node]:
            euler_tour(child)
            dfs_order.append(node)

    euler_tour(1)

    for node in dfs_order:
        while idx < len(insurances_sorted) and insurances_sorted[idx][0] == entry[node]:
            x_entry, max_depth = insurances_sorted[idx]
            heapq.heappush(pq, max_depth)
            idx += 1
        while pq and pq[0] < depth[node]:
            heapq.heappop(pq)
        if pq:
            covered[entry[node]] = True

    print(sum(covered[1:N+1]))

if __name__ == "__main__":
    main()