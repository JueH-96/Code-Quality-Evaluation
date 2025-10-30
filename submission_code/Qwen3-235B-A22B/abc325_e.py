import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    A = int(data[idx])
    idx +=1
    B = int(data[idx])
    idx +=1
    C = int(data[idx])
    idx +=1
    
    D = []
    for _ in range(N):
        row = list(map(int, data[idx:idx+N]))
        D.append(row)
        idx += N
    
    INF = 1 << 60
    adj = [[] for _ in range(2 * N + 2)]  # Nodes 1..N: car; N+1..2N: train
    
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j:
                continue
            # Car edges (i to j)
            cost_car = D[i-1][j-1] * A
            adj[i].append( (j, cost_car) )
            # Train edges (i+N to j+N)
            cost_train = D[i-1][j-1] * B + C
            adj[i + N].append( (j + N, cost_train) )
        # Switch edge from car to train at i
        adj[i].append( (i + N, 0) )
    
    dist = [INF] * (2 * N + 2)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, cost in adj[u]:
            if dist[v] > d + cost:
                dist[v] = d + cost
                heapq.heappush(heap, (dist[v], v))
    
    ans = min(dist[N], dist[N * 2])
    print(ans)

if __name__ == "__main__":
    main()