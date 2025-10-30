import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    trains = []
    for _ in range(M):
        l = int(data[idx])
        d = int(data[idx+1])
        k = int(data[idx+2])
        c = int(data[idx+3])
        A = int(data[idx+4])
        B = int(data[idx+5])
        idx += 6
        trains.append((l, d, k, c, A, B))
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for l, d, k, c, A, B in trains:
        adj[B].append((A, l, d, k, c))
    
    # Initialize distances
    dist = [-1] * (N+1)
    dist[N] = float('inf')
    
    # Priority queue (max-heap)
    heap = []
    heapq.heappush(heap, (-dist[N], N))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        current_dist = -current_dist
        if current_dist < dist[u]:
            continue
        for v, l, d, k, c in adj[u]:
            # Calculate the latest departure time from v to u
            # The latest departure time is the latest t such that t + c <= current_dist
            # t <= current_dist - c
            # Also, t must be in the sequence l, l+d, ..., l+(k-1)d
            # So, t_max = min(current_dist - c, l + (k-1)*d)
            t_max = min(current_dist - c, l + (k-1)*d)
            if t_max < l:
                continue
            # Find the largest t <= t_max in the sequence
            # t = l + m*d, where m is the largest integer such that l + m*d <= t_max
            m = (t_max - l) // d
            t = l + m * d
            if t < l:
                continue
            if t + c > current_dist:
                continue
            if dist[v] < t:
                dist[v] = t
                heapq.heappush(heap, (-dist[v], v))
    
    for i in range(1, N):
        if dist[i] == -1:
            print("Unreachable")
        else:
            print(dist[i])

if __name__ == "__main__":
    main()