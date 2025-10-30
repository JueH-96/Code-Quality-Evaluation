import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0

N = int(data[index])
index += 1

weights = list(map(int, data[index:index+N]))
index += N

intervals = []
for i in range(N):
    L = int(data[index])
    R = int(data[index+1])
    intervals.append((L, R, i))
    index += 2

Q = int(data[index])
index += 1

queries = []
for i in range(Q):
    s = int(data[index]) - 1
    t = int(data[index+1]) - 1
    queries.append((s, t))
    index += 2

# Sort intervals by their starting point
intervals.sort()

# Graph adjacency list
graph = defaultdict(list)

# To efficiently find overlapping intervals, we can use a sweep line algorithm
# with an active set of intervals, maintained in a min-heap by their end points.
active_intervals = []

i = 0
while i < N:
    L, R, idx = intervals[i]
    
    # Remove intervals from the active set that end before the current interval starts
    while active_intervals and active_intervals[0][0] < L:
        heapq.heappop(active_intervals)
    
    # All intervals in the active set do not overlap with the current interval
    for _, _, active_idx in active_intervals:
        graph[idx].append(active_idx)
        graph[active_idx].append(idx)
    
    # Add the current interval to the active set
    heapq.heappush(active_intervals, (R, L, idx))
    
    i += 1

# Function to find the minimum path weight using BFS
def min_path_weight(s, t):
    # BFS to find the shortest path from s to t
    queue = deque([(s, weights[s])])
    visited = [False] * N
    visited[s] = True
    
    while queue:
        current, current_weight = queue.popleft()
        
        if current == t:
            return current_weight
        
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, current_weight + weights[neighbor]))
    
    return -1

# Process each query
results = []
for s, t in queries:
    result = min_path_weight(s, t)
    results.append(result)

# Print all results
for result in results:
    print(result)