n = int(input())
s = input().strip()
t = input().strip()

# Check if transformation is possible and build mapping
s_to_positions = {}
for i in range(n):
    if s[i] not in s_to_positions:
        s_to_positions[s[i]] = []
    s_to_positions[s[i]].append(i)

s_to_t = {}
for s_char, positions in s_to_positions.items():
    t_chars = set(t[i] for i in positions)
    if len(t_chars) > 1:
        print(-1)
        exit()
    s_to_t[s_char] = t_chars.pop()

# Build the graph (only include edges where s_char != t_char)
graph = {}
for s_char, t_char in s_to_t.items():
    if s_char != t_char:
        graph[s_char] = t_char

# Find cycles and paths
visited = set()
operations = 0

for start in graph:
    if start in visited:
        continue
    
    # Trace the component starting from start
    path = []
    current = start
    
    while current not in visited:
        if current not in graph:
            # End of path (current doesn't need to change)
            break
        
        path.append(current)
        visited.add(current)
        current = graph[current]
    
    if current in path:
        # We found a cycle
        cycle_start_index = path.index(current)
        tail_length = cycle_start_index
        cycle_length = len(path) - cycle_start_index
        operations += tail_length + cycle_length + 1  # +1 for breaking the cycle
    else:
        # We found a path
        operations += len(path)

print(operations)