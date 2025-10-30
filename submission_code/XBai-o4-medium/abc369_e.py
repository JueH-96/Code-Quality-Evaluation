import sys
import heapq
from itertools import permutations

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    bridges = []
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        adj[u].append((v, t))
        adj[v].append((u, t))
        bridges.append((u, v, t))

    # Precompute all pairs shortest paths using Dijkstra for each node
    INF = float('inf')
    dist = [[INF] * (N + 1) for _ in range(N + 1)]

    for u in range(1, N + 1):
        dist_u = [INF] * (N + 1)
        dist_u[u] = 0
        heap = []
        heapq.heappush(heap, (0, u))
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current_dist > dist_u[current]:
                continue
            for neighbor, weight in adj[current]:
                if dist_u[neighbor] > current_dist + weight:
                    dist_u[neighbor] = current_dist + weight
                    heapq.heappush(heap, (dist_u[neighbor], neighbor))
        for v in range(1, N + 1):
            dist[u][v] = dist_u[v]

    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        K_i = int(input[ptr])
        ptr += 1
        B_list = list(map(int, input[ptr:ptr + K_i]))
        ptr += K_i
        query_bridges = [bridges[b - 1] for b in B_list]
        min_answer = INF

        for perm in permutations(query_bridges):
            current_positions = {1: 0}
            for bridge in perm:
                u, v, t = bridge
                new_positions = {}
                for prev_pos in current_positions:
                    cost_so_far = current_positions[prev_pos]
                    # Option 1: go to u, cross to v
                    d_prev_to_u = dist[prev_pos][u]
                    total_cost1 = cost_so_far + d_prev_to_u + t
                    if v in new_positions:
                        if total_cost1 < new_positions[v]:
                            new_positions[v] = total_cost1
                    else:
                        new_positions[v] = total_cost1
                    # Option 2: go to v, cross to u
                    d_prev_to_v = dist[prev_pos][v]
                    total_cost2 = cost_so_far + d_prev_to_v + t
                    if u in new_positions:
                        if total_cost2 < new_positions[u]:
                            new_positions[u] = total_cost2
                    else:
                        new_positions[u] = total_cost2
                current_positions = new_positions
            # After processing all bridges, compute the minimum total time
            current_min = INF
            for pos in current_positions:
                total = current_positions[pos] + dist[pos][N]
                if total < current_min:
                    current_min = total
            if current_min < min_answer:
                min_answer = current_min
        print(min_answer)

if __name__ == '__main__':
    main()