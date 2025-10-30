import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    roads = []
    adj = [[] for _ in range(N+1)]  # adjacency list

    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        roads.append((A, B, C))
        adj[A].append((B, C))
        adj[B].append((A, C))
    
    output = []
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        if query_type == '1':
            road_idx = int(input[ptr]) - 1  # convert to 0-based index
            ptr += 1
            A, B, C = roads[road_idx]
            # Remove (B, C) from adj[A]
            adj[A] = [(v, cost) for (v, cost) in adj[A] if v != B or cost != C]
            # Remove (A, C) from adj[B]
            adj[B] = [(v, cost) for (v, cost) in adj[B] if v != A or cost != C]
        else:
            x = int(input[ptr])
            ptr += 1
            y = int(input[ptr])
            ptr += 1
            # Dijkstra's algorithm
            INF = float('inf')
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            found = False
            while heap and not found:
                d, u = heapq.heappop(heap)
                if u == y:
                    found = True
                    break
                if d > dist[u]:
                    continue
                for (v, cost) in adj[u]:
                    if dist[v] > dist[u] + cost:
                        dist[v] = dist[u] + cost
                        heapq.heappush(heap, (dist[v], v))
            output.append(str(-1 if dist[y] == INF else dist[y]))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()