import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N+1)]  # 1-based indexing
    roads = []
    for i in range(1, M+1):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        adj[A].append((B, C, i))
        adj[B].append((A, C, i))
        roads.append((A, B, C, i))

    closed = set()

    for _ in range(Q):
        query_type = int(input[ptr])
        ptr += 1
        if query_type == 1:
            i = int(input[ptr])
            ptr += 1
            closed.add(i)
        else:
            x = int(input[ptr])
            ptr += 1
            y = int(input[ptr])
            ptr += 1

            INF = float('inf')
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))

            while heap:
                d_u, u = heapq.heappop(heap)
                if u == y:
                    break
                if d_u > dist[u]:
                    continue
                for v, c, road_i in adj[u]:
                    if road_i in closed:
                        continue
                    if dist[v] > dist[u] + c:
                        dist[v] = dist[u] + c
                        heapq.heappush(heap, (dist[v], v))
            
            if dist[y] == INF:
                print(-1)
            else:
                print(dist[y])

if __name__ == '__main__':
    main()