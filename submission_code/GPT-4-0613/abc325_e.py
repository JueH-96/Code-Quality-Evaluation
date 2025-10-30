import sys
import heapq

def dijkstra(s, n, adj):
    dist = [float('inf')] * n
    dist[s] = 0
    hq = [(0, s)]
    heapq.heapify(hq)
    while hq:
        v = heapq.heappop(hq)[1]
        for to, cost in adj[v]:
            if dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heapq.heappush(hq, (dist[to], to))
    return dist

def main():
    N, A, B, C = map(int, sys.stdin.readline().split())
    D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    adj = [[] for _ in range(2*N)]
    for i in range(N):
        for j in range(i+1, N):
            adj[i].append((j, D[i][j]*A))
            adj[j].append((i, D[i][j]*A))
            adj[i+N].append((j+N, D[i][j]*B+C))
            adj[j+N].append((i+N, D[i][j]*B+C))
        if i+1 < N:
            adj[i].append((i+1+N, 0))
    dist = dijkstra(0, 2*N, adj)
    print(min(dist[N-1], dist[2*N-1]))

if __name__ == "__main__":
    main()