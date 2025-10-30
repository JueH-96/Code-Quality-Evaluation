import sys
import heapq

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges[u].append((v, b, c))
    
    def can_achieve(ratio):
        dist = [float('-inf')] * (N+1)
        dist[1] = 0
        heap = []
        heapq.heappush(heap, (-dist[1], 1))
        while heap:
            current_dist, u = heapq.heappop(heap)
            current_dist = -current_dist
            if u == N:
                return True
            if current_dist < dist[u]:
                continue
            for v, b, c in edges[u]:
                new_dist = current_dist + (b - ratio * c)
                if new_dist > dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (-dist[v], v))
        return dist[N] >= 0
    
    low = 0
    high = 1e4
    for _ in range(100):
        mid = (low + high) / 2
        if can_achieve(mid):
            low = mid
        else:
            high = mid
    print("{0:.15f}".format(low))

if __name__ == "__main__":
    main()