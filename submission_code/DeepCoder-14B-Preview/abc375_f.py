import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    Q = int(input[ptr]); ptr += 1

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(1, M + 1):
        A = int(input[ptr]); ptr += 1
        B = int(input[ptr]); ptr += 1
        C = int(input[ptr]); ptr += 1
        adj[A].append((B, C, i))
        adj[B].append((A, C, i))

    closed = [False] * (M + 1)  # Using a list for O(1) access

    for _ in range(Q):
        query_type = input[ptr]; ptr += 1
        if query_type == '1':
            road_id = int(input[ptr]); ptr += 1
            closed[road_id] = True
        else:
            x = int(input[ptr]); ptr += 1
            y = int(input[ptr]); ptr += 1

            # Dijkstra's algorithm
            INF = float('inf')
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = [(0, x)]
            visited = [False] * (N + 1)
            found = False

            while heap:
                current_dist, u = heapq.heappop(heap)
                if u == y:
                    print(current_dist)
                    found = True
                    break
                if visited[u]:
                    continue
                visited[u] = True
                for v, c, rid in adj[u]:
                    if closed[rid]:
                        continue
                    if dist[v] > current_dist + c:
                        dist[v] = current_dist + c
                        heapq.heappush(heap, (dist[v], v))
            if not found:
                print(-1)

if __name__ == '__main__':
    main()