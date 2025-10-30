def count_reachable_pairs(N, a):
    visited = [False] * (N + 1)
    component_size = [0] * (N + 1)
    
    def dfs(start):
        current = start
        path = []
        while not visited[current]:
            visited[current] = True
            path.append(current)
            current = a[current - 1]
        
        # If we reach a visited node, we need to find the cycle
        if current in path:
            cycle_start_index = path.index(current)
            cycle_length = len(path) - cycle_start_index
            # Count the size of the cycle
            for i in range(cycle_start_index, len(path)):
                component_size[path[i]] = cycle_length
            
            # Count the size of the path leading to the cycle
            path_length = cycle_start_index
            for i in range(cycle_start_index):
                component_size[path[i]] = cycle_length + path_length
                path_length -= 1
        else:
            # If we reached a visited node that is not part of the current path
            for node in path:
                component_size[node] = 0  # Not part of any cycle
    
    # Perform DFS from each node
    for i in range(1, N + 1):
        if not visited[i]:
            dfs(i)
    
    # Calculate the total number of reachable pairs
    total_pairs = 0
    for size in component_size:
        if size > 0:
            total_pairs += size
    
    return total_pairs

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
a = list(map(int, data[1:N + 1]))

result = count_reachable_pairs(N, a)
print(result)