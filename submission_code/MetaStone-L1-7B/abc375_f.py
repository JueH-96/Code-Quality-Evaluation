import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]
    roads = []
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        roads.append((A, B, C))
        adj[A].append((B, len(roads) - 1, C))
        adj[B].append((A, len(roads) - 1, C))

    closed = set()

    output = []
    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            i = int(input[ptr]) - 1  # Convert to 0-based index
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
            visited = [False] * (N + 1)

            while heap:
                d, u = heapq.heappop(heap)
                if visited[u]:
                    continue
                visited[u] = True
                for v, road_idx, c in adj[u]:
                    if road_idx in closed:
                        continue
                    if dist[v] > d + c:
                        dist[v] = d + c
                        heapq.heappush(heap, (dist[v], v))

            if dist[y] == INF:
                output.append(-1)
            else:
                output.append(dist[y])

    print('
'.join(map(str, output)))

if __name__ == '__main__':
    main()