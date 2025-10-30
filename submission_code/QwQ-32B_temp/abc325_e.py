import sys
import heapq

def dijkstra(adj, start, N):
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[start] = 0
    heap = []
    heapq.heappush(heap, (0, start))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    A = int(input[idx]); idx +=1
    B = int(input[idx]); idx +=1
    C = int(input[idx]); idx +=1

    D = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            D[i][j] = int(input[idx]); idx +=1

    adj_car = [[] for _ in range(N+1)]
    adj_train = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        for v in range(1, N+1):
            if u != v:
                w_car = D[u][v] * A
                w_train = D[u][v] * B + C
                adj_car[u].append( (v, w_car) )
                adj_train[u].append( (v, w_train) )

    car_dist = dijkstra(adj_car, 1, N)
    train_dist = dijkstra(adj_train, N, N)

    min_total = float('inf')
    for k in range(1, N+1):
        total = car_dist[k] + train_dist[k]
        if total < min_total:
            min_total = total
    print(min_total)

if __name__ == "__main__":
    main()