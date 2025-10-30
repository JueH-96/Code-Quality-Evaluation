import sys
import heapq
from itertools import permutations, product

def dijkstra(adj, start, N):
    dist = [float('inf')] * (N + 1)
    dist[start] = 0
    heap = []
    heapq.heappush(heap, (0, start))
    visited = [False] * (N + 1)
    while heap:
        current_dist, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, _, t in adj[u]:
            if dist[v] > current_dist + t:
                dist[v] = current_dist + t
                heapq.heappush(heap, (dist[v], v))
    return dist

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    bridges = [None] * (M + 1)
    adj = [[] for _ in range(N + 1)]
    for i in range(1, M + 1):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        bridges[i] = (u, v, t)
        adj[u].append((v, i, t))
        adj[v].append((u, i, t))

    # Precompute all-pairs shortest paths
    dist_all = [[float('inf')] * (N + 1) for _ in range(N + 1)]
    for u in range(1, N + 1):
        dist = dijkstra(adj, u, N)
        for v in range(1, N + 1):
            dist_all[u][v] = dist[v]

    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        K_i = int(input[ptr])
        ptr += 1
        B_i = list(map(int, input[ptr:ptr + K_i]))
        ptr += K_i

        min_total = float('inf')
        # Generate all permutations of B_i
        for perm in permutations(B_i):
            # Generate all possible direction choices (2^K_i)
            for dirs in product([0, 1], repeat=len(perm)):
                current = 1
                total = 0
                valid = True
                for i in range(len(perm)):
                    b = perm[i]
                    u_b, v_b, t_b = bridges[b]
                    if dirs[i] == 0:
                        start = u_b
                        end = v_b
                    else:
                        start = v_b
                        end = u_b
                    # Find minimal distance from current to start
                    d = dist_all[current][start]
                    if d == float('inf'):
                        valid = False
                        break
                    total += d
                    total += t_b
                    current = end
                if not valid:
                    continue
                # Add distance from current to N
                d = dist_all[current][N]
                if d == float('inf'):
                    continue
                total += d
                if total < min_total:
                    min_total = total
        print(min_total)

if __name__ == '__main__':
    main()