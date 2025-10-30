import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
adj_list = defaultdict(list)

index = 2
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    edges.append((a, b))
    adj_list[a].append(b)
    index += 2

def bfs_find_cycle(start):
    queue = deque([(start, 0)])  # (current_node, distance_from_start)
    visited = {}
    min_cycle_length = float('inf')
    
    while queue:
        current, dist = queue.popleft()
        
        for neighbor in adj_list[current]:
            if neighbor == start:
                min_cycle_length = min(min_cycle_length, dist + 1)
            elif neighbor not in visited or dist + 1 < visited[neighbor]:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return min_cycle_length if min_cycle_length != float('inf') else -1

# We are only interested in cycles that contain vertex 1
result = bfs_find_cycle(1)

print(result)