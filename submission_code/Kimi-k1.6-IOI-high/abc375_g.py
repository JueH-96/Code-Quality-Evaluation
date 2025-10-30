import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]
    roads = []
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        c = int(input[idx])
        idx += 1
        adj[a].append((b, c))
        adj[b].append((a, c))
        roads.append((a, b, c))
    
    # Compute shortest paths from city 1
    d1 = [float('inf')] * (N + 1)
    d1[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    while heap:
        dist, u = heapq.heappop(heap)
        if dist > d1[u]:
            continue
        for v, cost in adj[u]:
            if d1[v] > d1[u] + cost:
                d1[v] = d1[u] + cost
                heapq.heappush(heap, (d1[v], v))
    
    # Compute shortest paths from city N (reverse is handled by undirected graph)
    d2 = [float('inf')] * (N + 1)
    d2[N] = 0
    heapq.heappush(heap, (0, N))
    while heap:
        dist, u = heapq.heappop(heap)
        if dist > d2[u]:
            continue
        for v, cost in adj[u]:
            if d2[v] > d2[u] + cost:
                d2[v] = d2[u] + cost
                heapq.heappush(heap, (d2[v], v))
    
    for a, b, c in roads:
        cond1 = (d1[a] + c + d2[b] == d1[N])
        cond2 = (d1[b] + c + d2[a] == d1[N])
        print("Yes" if cond1 or cond2 else "No")

if __name__ == '__main__':
    main()