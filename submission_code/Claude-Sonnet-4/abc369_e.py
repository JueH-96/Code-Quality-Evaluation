import heapq
from collections import defaultdict
import sys

def solve():
    N, M = map(int, input().split())
    
    # Store bridge information
    bridges = []
    graph = defaultdict(list)
    
    for i in range(M):
        u, v, t = map(int, input().split())
        bridges.append((u, v, t))
        graph[u].append((v, t, i))
        graph[v].append((u, t, i))
    
    Q = int(input())
    
    for _ in range(Q):
        line = list(map(int, input().split()))
        K = line[0]
        required_bridges = [x - 1 for x in line[1:]]  # Convert to 0-indexed
        
        # State: (cost, island, used_mask)
        # used_mask: bitmask indicating which required bridges have been used
        pq = [(0, 1, 0)]  # Start at island 1 with no bridges used
        visited = set()
        
        target_mask = (1 << K) - 1  # All required bridges used
        
        while pq:
            cost, island, used_mask = heapq.heappop(pq)
            
            if (island, used_mask) in visited:
                continue
            visited.add((island, used_mask))
            
            # Check if we reached island N with all required bridges used
            if island == N and used_mask == target_mask:
                print(cost)
                break
            
            # Try all adjacent bridges
            for next_island, bridge_cost, bridge_idx in graph[island]:
                new_cost = cost + bridge_cost
                new_mask = used_mask
                
                # Check if this bridge is one of the required bridges
                if bridge_idx in required_bridges:
                    bridge_pos = required_bridges.index(bridge_idx)
                    new_mask |= (1 << bridge_pos)
                
                if (next_island, new_mask) not in visited:
                    heapq.heappush(pq, (new_cost, next_island, new_mask))

solve()