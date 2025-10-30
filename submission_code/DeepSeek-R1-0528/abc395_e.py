import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    X = int(data[2])
    graph0 = [[] for _ in range(n+1)]
    graph1 = [[] for _ in range(n+1)]
    
    index = 3
    for _ in range(m):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        graph0[u].append(v)
        graph1[v].append(u)
    
    INF = 10**20
    dist = [[INF] * (n+1) for _ in range(2)]
    heap = []
    dist[0][1] = 0
    heapq.heappush(heap, (0, 0, 1))
    
    while heap:
        cost, s, u = heapq.heappop(heap)
        if cost != dist[s][u]:
            continue
            
        ns = 1 - s
        nc = cost + X
        if nc < dist[ns][u]:
            dist[ns][u] = nc
            heapq.heappush(heap, (nc, ns, u))
            
        if s == 0:
            for v in graph0[u]:
                if cost + 1 < dist[0][v]:
                    dist[0][v] = cost + 1
                    heapq.heappush(heap, (cost+1, 0, v))
        else:
            for v in graph1[u]:
                if cost + 1 < dist[1][v]:
                    dist[1][v] = cost + 1
                    heapq.heappush(heap, (cost+1, 1, v))
                    
    ans = min(dist[0][n], dist[1][n])
    print(ans)

if __name__ == '__main__':
    main()