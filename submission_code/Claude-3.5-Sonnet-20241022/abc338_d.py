def find_shortest_path(N, start, end, excluded_edge):
    if start == end:
        return 0
    
    # BFS to find shortest path
    visited = [False] * (N+1)
    queue = [(start, 0)]
    visited[start] = True
    
    while queue:
        curr, dist = queue.pop(0)
        
        # Try all possible next nodes
        for next_node in [(curr-1) if curr > 1 else N, (curr+1) if curr < N else 1]:
            # Skip if this edge is excluded
            if (min(curr,next_node), max(curr,next_node)) == excluded_edge:
                continue
                
            if not visited[next_node]:
                if next_node == end:
                    return dist + 1
                visited[next_node] = True
                queue.append((next_node, dist+1))
    
    return float('inf')

def solve_tour(N, M, X):
    min_length = float('inf')
    
    # Try removing each edge
    for edge in range(N):
        # edge i connects nodes i+1 and i+2 (mod N)
        excluded = (edge+1, (edge+2) if edge != N-1 else 1)
        
        # Calculate total path length for this configuration
        total_length = 0
        possible = True
        
        for i in range(M-1):
            length = find_shortest_path(N, X[i], X[i+1], excluded)
            if length == float('inf'):
                possible = False
                break
            total_length += length
            
        if possible:
            min_length = min(min_length, total_length)
            
    return min_length

# Read input
N, M = map(int, input().split())
X = list(map(int, input().split()))

# Print result
print(solve_tour(N, M, X))