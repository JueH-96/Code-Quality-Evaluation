import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx +=1
    A = list(map(int, data[idx:idx+n]))
    idx +=n
    adj = [[] for _ in range(n+1)]
    for _ in range(m):
        u = int(data[idx])
        idx +=1
        v = int(data[idx])
        idx +=1
        b = int(data[idx])
        idx +=1
        adj[u].append( (v, b + A[v-1]) )
        adj[v].append( (u, b + A[u-1]) )
    
    INF = float('inf')
    dist = [INF] * (n+1)
    dist[1] = A[0]
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    print(' '.join(map(str, dist[2:])))
    
if __name__ == "__main__":
    main()