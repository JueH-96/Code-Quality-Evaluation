import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

for i in range(1, 3 * (N - 1) + 1, 3):
    A = int(data[i])
    B = int(data[i+1])
    X = int(data[i+2])
    edges.append((A, B, X))

# Minimum time to reach each stage
min_time = [float('inf')] * (N + 1)
min_time[1] = 0

# Priority queue for processing stages
pq = [(0, 1)]  # (current_time, stage)

while pq:
    current_time, stage = heapq.heappop(pq)
    
    if stage == N:
        print(current_time)
        break
    
    if current_time > min_time[stage]:
        continue
    
    A, B, X = edges[stage - 1]
    
    # Option 1: Move to stage + 1
    next_stage = stage + 1
    next_time = current_time + A
    if next_time < min_time[next_stage]:
        min_time[next_stage] = next_time
        heapq.heappush(pq, (next_time, next_stage))
    
    # Option 2: Move to stage X
    next_stage = X
    next_time = current_time + B
    if next_time < min_time[next_stage]:
        min_time[next_stage] = next_time
        heapq.heappush(pq, (next_time, next_stage))