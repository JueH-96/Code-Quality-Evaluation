import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]  # 1-based indexing
    for _ in range(M):
        u = int(input[idx])
        idx +=1
        v = int(input[idx])
        idx +=1
        w = int(input[idx])
        idx +=1
        adj[u].append( (v, w) )
    
    INF = float('inf')
    max_mask = 1 << N
    full_mask = (1 << N) - 1
    # dist[mask][u] = minimal cost to reach u with visited nodes mask
    dist = [ [INF] * (N+1) for _ in range(max_mask) ]
    heap = []
    
    for u in range(1, N+1):
        mask = 1 << (u-1)
        dist[mask][u] = 0
        heapq.heappush(heap, (0, mask, u))
    
    found = False
    while heap:
        cost, mask, u = heapq.heappop(heap)
        if mask == full_mask:
            print(cost)
            found = True
            break
        if cost > dist[mask][u]:
            continue
        for v, w in adj[u]:
            new_mask = mask | (1 << (v-1))
            new_cost = cost + w
            if new_cost < dist[new_mask][v]:
                dist[new_mask][v] = new_cost
                heapq.heappush(heap, (new_cost, new_mask, v))
    
    if not found:
        print("No")

if __name__ == '__main__':
    main()