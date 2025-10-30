import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    adj = [[] for _ in range(N+1)]
    roads = []
    for i in range(1, M+1):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        roads.append( (A, B, C, i) )
        adj[A].append( (B, C, i) )
        adj[B].append( (A, C, i) )

    closed = [False] * (M+2)  # 1-based

    # Precompute initial APSP
    INF = float('inf')
    apsp = [ [INF] * (N+1) for _ in range(N+1) ]
    for u in range(1, N+1):
        dist = [INF] * (N+1)
        dist[u] = 0
        heap = [ (0, u) ]
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
            for to, c, idx in adj[v]:
                if closed[idx]:
                    continue
                if dist[to] > d + c:
                    dist[to] = d + c
                    heapq.heappush(heap, (dist[to], to))
        for v in range(1, N+1):
            apsp[u][v] = dist[v]

    for _ in range(Q):
        query = input[ptr]
        ptr +=1
        if query == '1':
            i = int(input[ptr])
            ptr +=1
            closed[i] = True
            # Update APSP for all nodes
            for u in range(1, N+1):
                dist = [INF] * (N+1)
                dist[u] = 0
                heap = [ (0, u) ]
                while heap:
                    d, v = heapq.heappop(heap)
                    if d > dist[v]:
                        continue
                    for to, c, idx in adj[v]:
                        if closed[idx]:
                            continue
                        if dist[to] > d + c:
                            dist[to] = d + c
                            heapq.heappush(heap, (dist[to], to))
                for v in range(1, N+1):
                    apsp[u][v] = dist[v]
        else:
            x = int(input[ptr]); ptr +=1
            y = int(input[ptr]); ptr +=1
            if apsp[x][y] == INF:
                print(-1)
            else:
                print(apsp[x][y])

if __name__ == '__main__':
    main()