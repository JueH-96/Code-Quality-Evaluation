import sys
import heapq

def main():
    data = sys.stdin.read().split()
    index = 0
    N = int(data[index])
    index += 1
    INF = 10**18
    adj = [[] for _ in range(N+1)]
    
    for i in range(1, N):
        A = int(data[index])
        B = int(data[index+1])
        X = int(data[index+2])
        index += 3
        adj[i].append((i+1, A))
        adj[i].append((X, B))
    
    distances = [INF] * (N+1)
    distances[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == N:
            break
        if current_dist > distances[u]:
            continue
        for v, w in adj[u]:
            if distances[v] > distances[u] + w:
                distances[v] = distances[u] + w
                heapq.heappush(heap, (distances[v], v))
    
    print(distances[N])

if __name__ == '__main__':
    main()