import sys
import heapq

def main():
    N = int(sys.stdin.readline())
    stages = []
    for _ in range(N-1):
        A, B, X = map(int, sys.stdin.readline().split())
        stages.append((A, B, X))
    
    # Initialize distance array
    INF = float('inf')
    dist = [INF] * (N+1)
    dist[1] = 0
    
    # Priority queue to manage the stages to process
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == N:
            break
        if current_dist > dist[u]:
            continue
        if u >= N:
            continue
        A, B, X = stages[u-1]
        # Option 1: Go to u+1
        if u+1 <= N:
            new_dist = dist[u] + A
            if new_dist < dist[u+1]:
                dist[u+1] = new_dist
                heapq.heappush(heap, (new_dist, u+1))
        # Option 2: Go to X
        if X <= N:
            new_dist = dist[u] + B
            if new_dist < dist[X]:
                dist[X] = new_dist
                heapq.heappush(heap, (new_dist, X))
    
    print(dist[N])

if __name__ == "__main__":
    main()