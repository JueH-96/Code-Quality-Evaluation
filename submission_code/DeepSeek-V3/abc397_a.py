import heapq

def solve():
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
    
    adj = [[] for _ in range(N+1)]
    rev_adj = [[] for _ in range(N+1)]
    
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        idx +=2
        adj[u].append(v)
        rev_adj[v].append(u)
    
    INF = 1 << 60
    dist = [[INF] * 2 for _ in range(N+1)]
    dist[1][0] = 0
    heap = []
    heapq.heappush(heap, (0, 1, 0))
    
    while heap:
        current_dist, u, state = heapq.heappop(heap)
        if u == N:
            print(current_dist)
            return
        if current_dist > dist[u][state]:
            continue
        # Move along edges in current state
        if state == 0:
            edges = adj[u]
        else:
            edges = rev_adj[u]
        for v in edges:
            if dist[v][state] > current_dist + 1:
                dist[v][state] = current_dist + 1
                heapq.heappush(heap, (dist[v][state], v, state))
        # Flip the state
        new_state = 1 - state
        if dist[u][new_state] > current_dist + X:
            dist[u][new_state] = current_dist + X
            heapq.heappush(heap, (dist[u][new_state], u, new_state))
    
    print(min(dist[N][0], dist[N][1]))

solve()