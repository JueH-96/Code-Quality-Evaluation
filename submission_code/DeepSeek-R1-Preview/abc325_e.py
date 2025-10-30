import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = int(data[idx])
    idx += 1
    B = int(data[idx])
    idx += 1
    C = int(data[idx])
    idx += 1
    
    D = []
    for _ in range(N):
        row = list(map(int, data[idx:idx+N]))
        D.append(row)
        idx += N
    
    INF = 1 << 60
    dist = [[INF] * 2 for _ in range(N + 1)]
    dist[1][0] = 0
    heap = []
    heapq.heappush(heap, (0, 1, 0))
    
    while heap:
        current_time, u, s = heapq.heappop(heap)
        if current_time > dist[u][s]:
            continue
        if u == N:
            continue
        for v in range(1, N + 1):
            if v == u:
                continue
            d = D[u - 1][v - 1]
            if s == 0:
                new_time_car = current_time + d * A
                if new_time_car < dist[v][0]:
                    dist[v][0] = new_time_car
                    heapq.heappush(heap, (new_time_car, v, 0))
                new_time_train = current_time + d * B + C
                if new_time_train < dist[v][1]:
                    dist[v][1] = new_time_train
                    heapq.heappush(heap, (new_time_train, v, 1))
            else:
                new_time_train = current_time + d * B + C
                if new_time_train < dist[v][1]:
                    dist[v][1] = new_time_train
                    heapq.heappush(heap, (new_time_train, v, 1))
    
    print(min(dist[N][0], dist[N][1]))

if __name__ == '__main__':
    main()