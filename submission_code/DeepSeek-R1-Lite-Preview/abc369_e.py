import sys
import heapq

def readints():
    return list(map(int, sys.stdin.readline().split()))

def dijkstra(N, adj, start):
    dist = [float('inf')] * (N + 1)
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist

def main():
    input = sys.stdin.read().splitlines()
    idx = 0
    N, M = map(int, input[idx].split())
    idx += 1
    adj = [[] for _ in range(N + 1)]
    bridges = []
    for _ in range(M):
        u, v, t = map(int, input[idx].split())
        adj[u].append((v, t))
        adj[v].append((u, t))
        bridges.append((u, v, t))
        idx += 1
    # Precompute all-pairs shortest paths
    all_pairs = [[0] * (N + 1) for _ in range(N + 1)]
    for u in range(1, N + 1):
        dist = dijkstra(N, adj, u)
        for v in range(1, N + 1):
            all_pairs[u][v] = dist[v]
    Q = int(input[idx])
    idx += 1
    for _ in range(Q):
        K = int(input[idx])
        idx += 1
        B = list(map(int, input[idx].split()))
        idx += 1
        # Collect required bridges and their endpoints
        required = []
        for b in B:
            u, v, t = bridges[b - 1]
            required.append((u, v, t))
        min_time = float('inf')
        # Generate all permutations of required bridges
        from itertools import permutations
        for perm in permutations(required):
            # For each permutation, try both starting directions
            for first_start in [0, 1]:
                current = 1
                total = 0
                # Choose starting point for first bridge
                if first_start == 0:
                    start = perm[0][0]
                    end = perm[0][1]
                else:
                    start = perm[0][1]
                    end = perm[0][0]
                # Go from 1 to start of first bridge
                total += all_pairs[1][start]
                # Traverse first bridge
                total += perm[0][2]
                current = end
                # Traverse remaining bridges
                for i in range(1, len(perm)):
                    bridge = perm[i]
                    # Choose starting point for this bridge
                    # To minimize, choose the closest starting point
                    if all_pairs[current][bridge[0]] < all_pairs[current][bridge[1]]:
                        total += all_pairs[current][bridge[0]]
                        current = bridge[1]
                    else:
                        total += all_pairs[current][bridge[1]]
                        current = bridge[0]
                    # Traverse the bridge
                    total += bridge[2]
                # Go from last endpoint to N
                total += all_pairs[current][N]
                if total < min_time:
                    min_time = total
        print(min_time)

if __name__ == "__main__":
    main()