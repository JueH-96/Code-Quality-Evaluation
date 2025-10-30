import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    # Build adjacency list with bridge indices
    adj = [[] for _ in range(N + 1)]  # 1-based indexing for nodes
    for i in range(1, M + 1):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        adj[u].append((v, t, i))
        adj[v].append((u, t, i))

    Q = int(input[ptr])
    ptr += 1

    for _ in range(Q):
        K = int(input[ptr])
        ptr += 1
        B = list(map(int, input[ptr:ptr + K]))
        ptr += K

        # Create mapping from bridge index to its position in the query's list
        bridge_to_pos = {b: idx for idx, b in enumerate(B)}
        K = len(B)
        size_mask = 1 << K
        INF = 1 << 60

        # Initialize distance array
        dist = [[INF] * size_mask for _ in range(N + 1)]
        dist[1][0] = 0
        heap = []
        heapq.heappush(heap, (0, 1, 0))
        found = False
        answer = INF

        while heap:
            time, u, mask = heapq.heappop(heap)
            if u == N and mask == size_mask - 1:
                answer = time
                found = True
                break
            if time > dist[u][mask]:
                continue
            for (v, t, idx) in adj[u]:
                new_mask = mask
                if idx in bridge_to_pos:
                    pos = bridge_to_pos[idx]
                    if not (mask & (1 << pos)):
                        new_mask = mask | (1 << pos)
                new_time = time + t
                if new_time < dist[v][new_mask]:
                    dist[v][new_mask] = new_time
                    heapq.heappush(heap, (new_time, v, new_mask))

        print(answer)

if __name__ == '__main__':
    main()