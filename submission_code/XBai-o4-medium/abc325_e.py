import heapq
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    A = int(input[idx]); idx += 1
    B = int(input[idx]); idx += 1
    C = int(input[idx]); idx += 1
    
    D = []
    for _ in range(N):
        row = list(map(int, input[idx:idx + N]))
        D.append(row)
        idx += N
    
    INF = 10**18
    
    # Dijkstra for car: starting from 0 (city 1)
    dist_car = [INF] * N
    dist_car[0] = 0
    heap = [(0, 0)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist_car[u]:
            continue
        for v in range(N):
            if u == v:
                continue
            weight = D[u][v] * A
            if dist_car[u] + weight < dist_car[v]:
                dist_car[v] = dist_car[u] + weight
                heapq.heappush(heap, (dist_car[v], v))
    
    # Dijkstra for train: starting from N-1 (city N)
    dist_train = [INF] * N
    dist_train[N-1] = 0
    heap = [(0, N-1)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist_train[u]:
            continue
        for v in range(N):
            if u == v:
                continue
            weight = D[u][v] * B + C
            if dist_train[u] + weight < dist_train[v]:
                dist_train[v] = dist_train[u] + weight
                heapq.heappush(heap, (dist_train[v], v))
    
    min_total = min(dist_car[k] + dist_train[k] for k in range(N))
    print(min_total)

if __name__ == "__main__":
    main()