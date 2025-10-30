import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    edges = [[] for _ in range(n+1)]  # 1-based indexing
    
    index = 1
    for i in range(1, n):
        a = int(data[index])
        b = int(data[index+1])
        x = int(data[index+2])
        index += 3
        edges[i].append((i+1, a))
        edges[i].append((x, b))
    
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == n:
            break
        if current_dist > dist[u]:
            continue
        for v, w in edges[u]:
            if dist[v] > current_dist + w:
                dist[v] = current_dist + w
                heapq.heappush(heap, (dist[v], v))
    
    print(dist[n])

if __name__ == "__main__":
    main()