import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    adj = [[] for _ in range(N+1)]  # 1-based indexing for nodes
    road_info = []

    for road_id in range(1, M+1):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        adj[A].append((B, C, road_id))
        adj[B].append((A, C, road_id))
        road_info.append((A, B, C))

    closed = [False] * (M + 2)  # road_id starts at 1
    output = []

    for _ in range(Q):
        t = int(input[ptr]); ptr +=1
        if t == 1:
            i = int(input[ptr]); ptr +=1
            closed[i] = True
        else:
            x = int(input[ptr]); ptr +=1
            y = int(input[ptr]); ptr +=1
            # Dijkstra's algorithm
            INF = float('inf')
            dist = [INF] * (N + 2)
            dist[x] = 0
            hq = []
            heapq.heappush(hq, (0, x))
            visited = [False] * (N + 2)
            found = False

            while hq:
                d, u = heapq.heappop(hq)
                if u == y:
                    output.append(str(d))
                    found = True
                    break
                if visited[u]:
                    continue
                visited[u] = True
                for v, c, rid in adj[u]:
                    if closed[rid]:
                        continue
                    if not visited[v] and dist[v] > d + c:
                        dist[v] = d + c
                        heapq.heappush(hq, (dist[v], v))
            if not found:
                output.append("-1")

    print('
'.join(output))

if __name__ == '__main__':
    main()