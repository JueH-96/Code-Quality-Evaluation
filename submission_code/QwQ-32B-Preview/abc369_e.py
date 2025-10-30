import sys
import heapq

def main():
    from sys import stdin
    input = stdin.read().split()
    idx = 0

    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    bridges = [None] * (M + 1)
    graph = [[] for _ in range(N + 1)]

    for b in range(1, M + 1):
        U = int(input[idx])
        idx += 1
        V = int(input[idx])
        idx += 1
        T = int(input[idx])
        idx += 1
        bridges[b] = (U, V, T)
        graph[U].append((V, b))
        graph[V].append((U, b))

    Q = int(input[idx])
    idx += 1

    for _ in range(Q):
        K_i = int(input[idx])
        idx += 1
        mandatory = list(map(int, input[idx:idx + K_i]))
        idx += K_i
        bridge_to_bit = {bridge: bit for bit, bridge in enumerate(mandatory)}

        dp = [[float('inf')] * (1 << K_i) for _ in range(N + 1)]
        dp[1][0] = 0

        pq = [(0, 1, 0)]
        visited = set()

        while pq:
            time, u, mask = heapq.heappop(pq)
            if (u, mask) in visited:
                continue
            visited.add((u, mask))

            if u == N and mask == (1 << K_i) - 1:
                print(time)
                break

            for v, b in graph[u]:
                new_mask = mask
                if b in bridge_to_bit:
                    bit = bridge_to_bit[b]
                    if not (mask & (1 << bit)):
                        new_mask |= (1 << bit)
                new_time = time + bridges[b][2]
                if dp[v][new_mask] > new_time:
                    dp[v][new_mask] = new_time
                    heapq.heappush(pq, (new_time, v, new_mask))

if __name__ == "__main__":
    main()