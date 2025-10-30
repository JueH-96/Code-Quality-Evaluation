import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    index = 1
    for i in range(1, n):
        a = int(data[index])
        b = int(data[index+1])
        x = int(data[index+2])
        index += 3
        graph[i].append((i+1, a))
        graph[i].append((x, b))
    
    INF = 10**18
    dist = [INF] * (n+1)
    dist[1] = 0
    heap = [(0, 1)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        for neighbor, w in graph[node]:
            nd = d + w
            if nd < dist[neighbor]:
                dist[neighbor] = nd
                heapq.heappush(heap, (nd, neighbor))
                
    print(dist[n])

if __name__ == "__main__":
    main()