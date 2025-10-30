import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        b = int(data[idx+2])
        edges[u].append((v, b))
        edges[v].append((u, b))
        idx += 3
    dist = [float('inf')] * (N+1)
    dist[1] = A[0]
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, b in edges[u]:
            new_dist = dist[u] + b + A[v-1]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    result = []
    for i in range(2, N+1):
        result.append(str(dist[i]))
    print(' '.join(result))

if __name__ == "__main__":
    main()