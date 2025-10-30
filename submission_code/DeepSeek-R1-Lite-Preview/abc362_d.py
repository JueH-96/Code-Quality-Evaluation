import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = [0] * (N + 1)
    index = 2
    for i in range(1, N + 1):
        A[i] = int(data[index])
        index += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        U = int(data[index])
        V = int(data[index + 1])
        B = int(data[index + 2])
        index += 3
        adj[U].append((V, B + A[V]))
        adj[V].append((U, B + A[U]))
    
    INF = 1 << 60
    dist = [INF] * (N + 1)
    dist[1] = A[1]
    
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    result = []
    for i in range(2, N + 1):
        result.append(str(dist[i]))
    
    print(' '.join(result))

if __name__ == '__main__':
    main()