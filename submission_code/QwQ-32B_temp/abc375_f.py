import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    Q = int(input[idx]); idx +=1

    adj = [[] for _ in range(N+1)]  # 1-based
    roads = []
    for i in range(1, M+1):
        a = int(input[idx]); idx +=1
        b = int(input[idx]); idx +=1
        c = int(input[idx]); idx +=1
        adj[a].append( (b, c, i) )
        adj[b].append( (a, c, i) )
        roads.append( (a, b, c) )  # Not used, but stored for reference

    closed = [False]*(M+1)  # 1-based indexing for roads

    for _ in range(Q):
        query_type = input[idx]; idx +=1
        if query_type == '1':
            i = int(input[idx]); idx +=1
            closed[i] = True
        else:
            x = int(input[idx]); idx +=1
            y = int(input[idx]); idx +=1
            INF = float('inf')
            dist = [INF]*(N+1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            while heap:
                d, u = heapq.heappop(heap)
                if u == y:
                    break
                if d > dist[u]:
                    continue
                for edge in adj[u]:
                    v, c, idx_edge = edge
                    if not closed[idx_edge]:
                        if dist[v] > d + c:
                            dist[v] = d + c
                            heapq.heappush(heap, (dist[v], v))
            if dist[y] == INF:
                print(-1)
            else:
                print(dist[y])

if __name__ == "__main__":
    main()