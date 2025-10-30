import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    A = int(input[ptr])
    ptr +=1
    B = int(input[ptr])
    ptr +=1
    C = int(input[ptr])
    ptr +=1
    
    D = []
    for _ in range(N):
        row = list(map(int, input[ptr:ptr+N]))
        D.append(row)
        ptr += N
    
    INF = float('inf')
    size = 2 * N
    dist = [INF] * size
    dist[0] = 0  # city 0, car mode
    heap = []
    heapq.heappush(heap, (0, 0))
    
    target_city = N-1
    target_car = 2 * target_city
    target_train = 2 * target_city + 1
    
    while heap:
        time, u = heapq.heappop(heap)
        if u == target_car or u == target_train:
            # Once we reach the target, we can still have shorter paths coming later, but Dijkstra's ensures the first time is the shortest
            # However, we need to process all possibilities to ensure we find the minimum
            pass
        if time > dist[u]:
            continue
        
        if u % 2 == 0:  # car mode
            city = u // 2
            for j in range(N):
                # Car to car
                v = 2 * j
                new_time = time + D[city][j] * A
                if new_time < dist[v]:
                    dist[v] = new_time
                    heapq.heappush(heap, (new_time, v))
                # Car to train
                v = 2 * j + 1
                new_time = time + D[city][j] * B + C
                if new_time < dist[v]:
                    dist[v] = new_time
                    heapq.heappush(heap, (new_time, v))
        else:  # train mode
            city = (u -1) // 2
            for j in range(N):
                v = 2 * j + 1
                new_time = time + D[city][j] * B + C
                if new_time < dist[v]:
                    dist[v] = new_time
                    heapq.heappush(heap, (new_time, v))
    
    print(min(dist[target_car], dist[target_train]))

if __name__ == "__main__":
    main()