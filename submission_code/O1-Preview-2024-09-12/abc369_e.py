# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import heapq
    sys.setrecursionlimit(1 << 25)

    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]  # 1-based indexing
    edges = []

    for idx in range(1, M+1):
        U_i, V_i, T_i = map(int, sys.stdin.readline().split())
        # Build adjacency list with edge information
        adj[U_i].append((V_i, T_i, idx))
        adj[V_i].append((U_i, T_i, idx))

    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        K_i = int(sys.stdin.readline())
        B_i = list(map(int, sys.stdin.readline().split()))
        queries.append((K_i, B_i))

    for query in queries:
        K_i, B_i = query
        # Map bridge index to bit position
        bridge_to_bit = {}
        for pos, bridge_idx in enumerate(B_i):
            bridge_to_bit[bridge_idx] = pos

        # Initialize distances: dist[node][mask]
        max_mask = 1 << K_i
        NODES = N + 1  # 1-based indexing
        INF = float('inf')
        dist = [ [INF] * max_mask for _ in range(NODES) ]
        dist[1][0] = 0
        heap = []
        heapq.heappush(heap, (0, 1, 0))  # (distance, node, mask)

        while heap:
            curr_dist, u, mask = heapq.heappop(heap)
            if dist[u][mask] < curr_dist:
                continue
            # Early termination if we've reached node N and used all bridges
            if u == N and mask == max_mask - 1:
                break
            for v, time, bridge_idx in adj[u]:
                new_mask = mask
                if bridge_idx in bridge_to_bit:
                    bit_pos = bridge_to_bit[bridge_idx]
                    new_mask = mask | (1 << bit_pos)
                if dist[v][new_mask] > curr_dist + time:
                    dist[v][new_mask] = curr_dist + time
                    heapq.heappush(heap, (dist[v][new_mask], v, new_mask))
        answer = dist[N][max_mask -1]
        # If answer is still INF, it means it's impossible
        print(int(answer))

threading.Thread(target=main).start()