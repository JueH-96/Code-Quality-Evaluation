from collections import defaultdict
import heapq

def solve():
    N = int(input())
    graph = defaultdict(list)
    
    # Build graph where each stage has edges to next stages it can reach
    for i in range(1, N):
        a, b, x = map(int, input().split())
        # For each stage i, we can go to i+1 with cost a or x_i with cost b
        graph[i].append((i+1, a))
        graph[i].append((x, b))

    # Dijkstra's algorithm to find shortest path from stage 1 to N
    dist = [float('inf')] * (N+1)
    dist[1] = 0
    pq = [(0, 1)]  # (distance, stage)
    
    while pq:
        d, curr = heapq.heappop(pq)
        
        if curr == N:
            print(d)
            return
            
        if d > dist[curr]:
            continue
            
        for next_stage, cost in graph[curr]:
            new_dist = d + cost
            if new_dist < dist[next_stage]:
                dist[next_stage] = new_dist
                heapq.heappush(pq, (new_dist, next_stage))

solve()