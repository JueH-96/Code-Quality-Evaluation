import sys
input = sys.stdin.read

def find_min_operations(N, M, sets):
    from collections import defaultdict, deque

    # Create a graph where each node represents a set and edges represent common elements
    graph = defaultdict(list)
    set_indices = {}

    for i in range(N):
        set_indices[i] = set(sets[i])
        for j in range(i + 1, N):
            if set_indices[i] & set_indices[j]:  # Check if there is a common element
                graph[i].append(j)
                graph[j].append(i)

    # BFS to find the minimum number of operations
    queue = deque()
    visited = set()
    steps = -1

    # Find all sets containing 1
    for i in range(N):
        if 1 in set_indices[i]:
            queue.append(i)
            visited.add(i)

    while queue:
        steps += 1
        for _ in range(len(queue)):
            current = queue.popleft()
            if M in set_indices[current]:
                return steps
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return -1

# Read input
data = input().strip().split()
N = int(data[0])
M = int(data[1])
index = 2
sets = []

for i in range(N):
    A = int(data[index])
    index += 1
    set_elements = list(map(int, data[index:index + A]))
    index += A
    sets.append(set_elements)

# Find and print the result
result = find_min_operations(N, M, sets)
print(result)