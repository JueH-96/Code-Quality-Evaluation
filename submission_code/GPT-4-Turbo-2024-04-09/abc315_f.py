import sys
import math
from heapq import heappop, heappush

input = sys.stdin.read
data = input().split()

N = int(data[0])
coordinates = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Use a priority queue to implement Dijkstra's algorithm
# Each node in the priority queue: (cost, index, skipped_count)
pq = []
heappush(pq, (0, 0, 0))  # Starting at checkpoint 1 with cost 0 and 0 skips

# Minimum cost to reach each checkpoint with a given number of skips
min_cost = [[float('inf')] * N for _ in range(N)]
min_cost[0][0] = 0

while pq:
    current_cost, current_index, skip_count = heappop(pq)
    
    if current_index == N - 1:
        # If we reach the last checkpoint, we can output the result
        print(current_cost)
        break
    
    # Explore next checkpoints
    for next_index in range(current_index + 1, N):
        # Calculate the penalty for skipping checkpoints
        skips = next_index - current_index - 1
        penalty = (2 ** skips - 1) if skips > 0 else 0
        distance = euclidean_distance(*coordinates[current_index], *coordinates[next_index])
        new_cost = current_cost + distance + penalty
        
        # If this path is better, consider it
        if new_cost < min_cost[next_index][skips]:
            min_cost[next_index][skips] = new_cost
            heappush(pq, (new_cost, next_index, skips))