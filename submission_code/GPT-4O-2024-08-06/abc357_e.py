def count_reachable_pairs(N, a):
    # Convert 1-based index to 0-based for easier handling in Python
    a = [x - 1 for x in a]
    
    visited = [False] * N
    cycle_size = [0] * N
    in_cycle = [False] * N
    cycle_id = [-1] * N
    
    def find_cycle(start):
        path = []
        current = start
        while not visited[current]:
            visited[current] = True
            path.append(current)
            current = a[current]
        
        # Find the start of the cycle in the path
        if current in path:
            cycle_start_index = path.index(current)
            cycle = path[cycle_start_index:]
            cycle_length = len(cycle)
            for node in cycle:
                in_cycle[node] = True
                cycle_size[node] = cycle_length
                cycle_id[node] = current
            return cycle_length
        return 0
    
    # Find all cycles
    for i in range(N):
        if not visited[i]:
            find_cycle(i)
    
    # Calculate reachability
    reach_count = 0
    for i in range(N):
        if in_cycle[i]:
            reach_count += cycle_size[i]
        else:
            # Traverse to find the cycle this node leads into
            current = i
            while not in_cycle[current]:
                current = a[current]
            # Add the cycle size to the reach count
            reach_count += cycle_size[current]
    
    return reach_count

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
a = list(map(int, data[1:]))

# Output the result
print(count_reachable_pairs(N, a))