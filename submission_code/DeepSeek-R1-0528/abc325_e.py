import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = int(data[1])
    B = int(data[2])
    C = int(data[3])
    idx = 4
    D = []
    for i in range(n):
        row = list(map(int, data[idx:idx+n]))
        D.append(row)
        idx += n

    INF = 10**25
    dist0 = [INF] * n
    heap = []
    dist0[0] = 0
    heapq.heappush(heap, (0, 0))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist0[u]:
            continue
        for v in range(n):
            if u == v:
                continue
            time_car = d + D[u][v] * A
            if time_car < dist0[v]:
                dist0[v] = time_car
                heapq.heappush(heap, (dist0[v], v))
                
    dist1 = [INF] * n
    heap = []
    for i in range(n):
        dist1[i] = dist0[i]
        heapq.heappush(heap, (dist1[i], i))
        
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist1[u]:
            continue
        for v in range(n):
            if u == v:
                continue
            time_train = d + D[u][v] * B + C
            if time_train < dist1[v]:
                dist1[v] = time_train
                heapq.heappush(heap, (dist1[v], v))
                
    print(dist1[n-1])

if __name__ == '__main__':
    main()