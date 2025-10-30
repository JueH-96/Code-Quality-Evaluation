import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2

sets = []
element_to_sets = defaultdict(list)

for i in range(N):
    A_i = int(data[index])
    index += 1
    current_set = set(map(int, data[index:index + A_i]))
    sets.append(current_set)
    for element in current_set:
        element_to_sets[element].append(i)
    index += A_i

# To find the connected components, we will use a union-find (disjoint set) structure
parent = list(range(N))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rootX != rootY:
        parent[rootY] = rootX

# Union sets that share common elements
for indices in element_to_sets.values():
    for i in range(1, len(indices)):
        union(indices[0], indices[i])

# Find all sets that contain 1 and M
sets_containing_1 = set()
sets_containing_M = set()

for i in range(N):
    if 1 in sets[i]:
        sets_containing_1.add(find(i))
    if M in sets[i]:
        sets_containing_M.add(find(i))

# If there's any intersection between the sets containing 1 and M, it's possible
if sets_containing_1 & sets_containing_M:
    # We need to find the minimum number of operations
    # This is equivalent to finding the minimum number of edges in the component graph
    # that connects all nodes in the component containing both 1 and M
    component = next(iter(sets_containing_1 & sets_containing_M))
    
    # BFS to find the shortest path from any node containing 1 to any node containing M
    queue = deque()
    visited = set()
    for i in range(N):
        if find(i) == component:
            if 1 in sets[i]:
                queue.append((i, 0))
                visited.add(i)
    
    while queue:
        current, steps = queue.popleft()
        if M in sets[current]:
            print(steps)
            break
        
        for element in sets[current]:
            for neighbor in element_to_sets[element]:
                if find(neighbor) == component and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))
else:
    print(-1)