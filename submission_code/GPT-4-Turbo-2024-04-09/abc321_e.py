def count_vertices_at_distance_k(N, X, K):
    if K == 0:
        return 1  # Distance from X to X is 0, so only one vertex (X itself)
    
    # Calculate the number of vertices at each level of the tree
    # Level 0 has 1 vertex (node 1)
    # Level 1 has 2 vertices (nodes 2, 3) if they exist
    # Level 2 has 4 vertices (nodes 4, 5, 6, 7) if they exist
    # and so on...
    
    # Find the level of vertex X
    level_X = 0
    current = X
    while current > 1:
        current //= 2
        level_X += 1
    
    # We need vertices at level (level_X + K) and (level_X - K)
    level_down = level_X + K
    level_up = level_X - K
    
    # Function to count vertices at a specific level
    def vertices_at_level(level):
        if level < 0:
            return 0
        # The maximum vertex number at this level if fully populated
        max_vertex_at_level = min((1 << (level + 1)) - 1, N)
        # The minimum vertex number at this level
        min_vertex_at_level = (1 << level)
        if min_vertex_at_level > N:
            return 0
        return max_vertex_at_level - min_vertex_at_level + 1
    
    count = 0
    # Count vertices at level_down
    count += vertices_at_level(level_down)
    
    # Count vertices at level_up (only if K != 0 to avoid double counting the same level)
    if K != 0:
        count += vertices_at_level(level_up)
    
    return count

import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3
    result = count_vertices_at_distance_k(N, X, K)
    results.append(result)

for result in results:
    print(result)