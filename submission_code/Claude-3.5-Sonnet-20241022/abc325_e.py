def dijkstra(N, A, B, C, D):
    import heapq
    INF = float('inf')
    
    # dist[i][j]: minimum time to reach city i using state j
    # j=0: can use both car and train, j=1: can only use train
    dist = [[INF]*2 for _ in range(N)]
    dist[0][0] = 0
    
    # (distance, city, state)
    q = [(0, 0, 0)]
    
    while q:
        d, v, state = heapq.heappop(q)
        
        if d > dist[v][state]:
            continue
            
        for u in range(N):
            if v == u:
                continue
                
            if state == 0:  # can use both car and train
                # Use car
                new_d = d + D[v][u] * A
                if new_d < dist[u][0]:
                    dist[u][0] = new_d
                    heapq.heappush(q, (new_d, u, 0))
                
                # Use train
                new_d = d + D[v][u] * B + C
                if new_d < dist[u][1]:
                    dist[u][1] = new_d
                    heapq.heappush(q, (new_d, u, 1))
                    
            else:  # can only use train
                new_d = d + D[v][u] * B + C
                if new_d < dist[u][1]:
                    dist[u][1] = new_d
                    heapq.heappush(q, (new_d, u, 1))
    
    return min(dist[N-1])

def main():
    N, A, B, C = map(int, input().split())
    D = []
    for _ in range(N):
        row = list(map(int, input().split()))
        D.append(row)
    
    ans = dijkstra(N, A, B, C, D)
    print(ans)

main()