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
        trains.append((A, B, l, d, k, c))
        idx += 6
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for train in trains:
        A, B, l, d, k, c = train
        adj[B].append((A, l, d, k, c))
    
    # Initialize distances
    dist = [ -1 ] * (N+1)
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
            # The latest arrival time at u is current_dist
            # The latest departure time from v is current_dist - c
            latest_departure = current_dist - c
            # Find the latest train from v to u that departs <= latest_departure
            # The train schedule is l, l+d, l+2d, ..., l+(k-1)d
            # We need to find the largest t <= latest_departure where t is in the schedule
            # t = l + x * d, x is integer, 0 <= x < k
            # t <= latest_departure
            # x <= (latest_departure - l) / d
            x_max = (latest_departure - l) // d
            x_max = min(x_max, k-1)
            if x_max < 0:
                continue
            t = l + x_max * d
            if t > latest_departure:
                continue
            # Update the distance for v
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