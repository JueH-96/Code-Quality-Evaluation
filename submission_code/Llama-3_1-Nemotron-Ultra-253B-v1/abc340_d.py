import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(n + 1)]  # 1-based indexing
    
    for i in range(1, n):
        a = int(data[idx])
        b = int(data[idx + 1])
        x = int(data[idx + 2])
        idx += 3
        adj[i].append((i + 1, a))
        adj[i].append((x, b))
    
    dist = [float('inf')] * (n + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if u == n:
            print(d)
            return
        if d > dist[u]:
            continue
        for v, cost in adj[u]:
            if dist[v] > d + cost:
                dist[v] = d + cost
                heapq.heappush(heap, (dist[v], v))
    
    # In case there's no path (which shouldn't happen according to the problem statement)
    print(-1)

if __name__ == "__main__":
    main()