import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]
    for bridge_idx in range(1, M + 1):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        adj[u].append((v, t, bridge_idx))
        adj[v].append((u, t, bridge_idx))

    Q = int(input[ptr])
    ptr += 1

    for _ in range(Q):
        K_i = int(input[ptr])
        ptr += 1
        B = list(map(int, input[ptr:ptr + K_i]))
        ptr += K_i

        required_bridges = B
        K = K_i
        full_mask = (1 << K) - 1
        bridge_to_index = {b: i for i, b in enumerate(required_bridges)}
        S = set(required_bridges)

        INF = float('inf')
        dist = [[INF] * (1 << K) for _ in range(N + 1)]
        dist[1][0] = 0
        heap = []
        heapq.heappush(heap, (0, 1, 0))

        found = False
        while heap:
            time, u, mask = heapq.heappop(heap)
            if u == N and mask == full_mask:
                print(time)
                found = True
                break
            if time > dist[u][mask]:
                continue
            for v, t, bridge_idx in adj[u]:
                new_mask = mask
                if bridge_idx in S:
                    idx = bridge_to_index[bridge_idx]
                    new_mask = mask | (1 << idx)
                new_time = time + t
                if new_time < dist[v][new_mask]:
                    dist[v][new_mask] = new_time
                    heapq.heappush(heap, (new_time, v, new_mask))
        if not found:
            print(-1)

if __name__ == "__main__":
    main()