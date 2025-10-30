import sys
import heapq

def main():
    N = int(sys.stdin.readline())
    stages = []
    for _ in range(N-1):
        A, B, X = map(int, sys.stdin.readline().split())
        stages.append((A, B, X))
    
    # Initialize distance array
    dist = [float('inf')] * (N+1)
    dist[1] = 0
    
    # Priority queue to manage the stages
    pq = []
    heapq.heappush(pq, (0, 1))
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if u == N:
            break
        if current_dist > dist[u]:
            continue
        if u < N:
            A, B, X = stages[u-1]
            # Option 1: Go to u+1
            if dist[u+1] > dist[u] + A:
                dist[u+1] = dist[u] + A
                heapq.heappush(pq, (dist[u+1], u+1))
            # Option 2: Go to X
            if dist[X] > dist[u] + B:
                dist[X] = dist[u] + B
                heapq.heappush(pq, (dist[X], X))
    
    print(dist[N])

if __name__ == "__main__":
    main()