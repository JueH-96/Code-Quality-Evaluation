import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    A = int(input[idx])
    idx += 1
    B = int(input[idx])
    idx += 1
    C = int(input[idx])
    idx += 1
    
    D = []
    for _ in range(n):
        row = list(map(int, input[idx:idx+n]))
        D.append(row)
        idx += n
    
    INF = float('inf')
    dist = [[INF] * 2 for _ in range(n)]
    dist[0][0] = 0  # Starting at city 0 (1-based 1), mode 0 (car)
    
    heap = []
    heapq.heappush(heap, (0, 0, 0))  # (time, city, mode)
    
    while heap:
        current_time, u, m = heapq.heappop(heap)
        if current_time > dist[u][m]:
            continue
        for v in range(n):
            if m == 0:
                time_added = D[u][v] * A
                new_time = current_time + time_added
                # Option 1: drive to v by car, still in car mode
                if new_time < dist[v][0]:
                    dist[v][0] = new_time
                    heapq.heappush(heap, (new_time, v, 0))
                # Option 2: drive to v by car, then switch to train
                new_time_switch = current_time + time_added
                if new_time_switch < dist[v][1]:
                    dist[v][1] = new_time_switch
                    heapq.heappush(heap, (new_time_switch, v, 1))
            else:
                # In train mode
                time_added = D[u][v] * B + C
                new_time = current_time + time_added
                if new_time < dist[v][1]:
                    dist[v][1] = new_time
                    heapq.heappush(heap, (new_time, v, 1))
    
    result = min(dist[n-1][0], dist[n-1][1])
    print(result)

if __name__ == '__main__':
    main()