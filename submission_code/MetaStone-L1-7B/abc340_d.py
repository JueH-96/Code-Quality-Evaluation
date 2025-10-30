import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    n = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(n + 1)]  # 1-based indexing
    
    for i in range(1, n):
        a = int(data[idx])
        b = int(data[idx + 1])
        x = int(data[idx + 2])
        idx += 3
        edges[i].append((i + 1, a))
        edges[i].append((x, b))
    
    # Dijkstra's algorithm
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == n:
            print(current_dist)
            return
        if current_dist > dist[u]:
            continue
        for v, cost in edges[u]:
            if dist[v] > current_dist + cost:
                dist[v] = current_dist + cost
                heapq.heappush(heap, (dist[v], v))
    
    print(dist[n])

if __name__ == '__main__':
    main()