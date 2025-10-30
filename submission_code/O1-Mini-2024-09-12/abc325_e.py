import sys, heapq

def main():
    import sys
    import sys
    from sys import stdin
    def input():
        return sys.stdin.read()
    
    data = list(map(int, input().split()))
    idx = 0
    N, A, B, C = data[idx], data[idx+1], data[idx+2], data[idx+3]
    idx +=4
    D = []
    for _ in range(N):
        row = data[idx:idx+N]
        D.append(row)
        idx +=N
    
    INF = 1 << 60
    dist = [ [INF]*N for _ in range(2) ]
    dist[0][0] = 0
    heap = []
    heapq.heappush(heap, (0, 0, 0))  # (distance, city, mode)
    
    while heap:
        current_dist, u, mode = heapq.heappop(heap)
        if u == N-1:
            print(current_dist)
            return
        if current_dist > dist[mode][u]:
            continue
        for v in range(N):
            if v == u:
                continue
            if mode == 0:
                # Take car
                new_dist = current_dist + D[u][v] * A
                if new_dist < dist[0][v]:
                    dist[0][v] = new_dist
                    heapq.heappush(heap, (new_dist, v, 0))
                # Switch to train
                new_dist = current_dist + D[u][v] * B + C
                if new_dist < dist[1][v]:
                    dist[1][v] = new_dist
                    heapq.heappush(heap, (new_dist, v, 1))
            else:
                # Use train only
                new_dist = current_dist + D[u][v] * B + C
                if new_dist < dist[1][v]:
                    dist[1][v] = new_dist
                    heapq.heappush(heap, (new_dist, v, 1))
    # If N-1 is not reachable
    print(-1)

if __name__ == "__main__":
    main()