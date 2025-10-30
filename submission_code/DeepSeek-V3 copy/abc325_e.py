import sys
import heapq

def main():
    N, A, B, C = map(int, sys.stdin.readline().split())
    D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Initialize distances
    dist = [float('inf')] * (2 * N)
    dist[0] = 0  # Starting at city 1 with car
    
    # Priority queue: (distance, node)
    # node: 0 to N-1 for car, N to 2N-1 for train
    pq = [(0, 0)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        if u < N:
            # Currently using car
            for v in range(N):
                if v == u:
                    continue
                # Continue using car
                new_dist = current_dist + D[u][v] * A
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
                # Switch to train
                new_dist = current_dist + D[u][v] * B + C
                if new_dist < dist[v + N]:
                    dist[v + N] = new_dist
                    heapq.heappush(pq, (new_dist, v + N))
        else:
            # Currently using train
            u_city = u - N
            for v in range(N):
                if v == u_city:
                    continue
                # Continue using train
                new_dist = current_dist + D[u_city][v] * B + C
                if new_dist < dist[v + N]:
                    dist[v + N] = new_dist
                    heapq.heappush(pq, (new_dist, v + N))
    
    # The destination is city N-1 (0-based)
    # We need to consider both car and train
    min_time = min(dist[N-1], dist[2*N-1])
    print(min_time)

if __name__ == "__main__":
    main()