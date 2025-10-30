import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    X = int(data[idx])
    idx += 1
    
    adj0 = [[] for _ in range(N+1)]
    adj1 = [[] for _ in range(N+1)]
    
    for _ in range(M):
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        adj0[u].append(v)
        adj1[v].append(u)
    
    INF = 1 << 60
    dist = [[INF] * 2 for _ in range(N+1)]
    dist[1][0] = 0
    heap = []
    heapq.heappush(heap, (0, 1, 0))
    
    while heap:
        cost, u, s = heapq.heappop(heap)
        if u == N:
            print(cost)
            return
        if cost > dist[u][s]:
            continue
        
        # Move along edges in current state
        if s == 0:
            neighbors = adj0[u]
        else:
            neighbors = adj1[u]
        
        for v in neighbors:
            new_cost = cost + 1
            if dist[v][s] > new_cost:
                dist[v][s] = new_cost
                heapq.heappush(heap, (new_cost, v, s))
        
        # Reverse edges
        new_s = 1 - s
        new_cost_rev = cost + X
        if dist[u][new_s] > new_cost_rev:
            dist[u][new_s] = new_cost_rev
            heapq.heappush(heap, (new_cost_rev, u, new_s)
    
    # As per problem statement, unreachable
    print(-1)

if __name__ == "__main__":
    main()