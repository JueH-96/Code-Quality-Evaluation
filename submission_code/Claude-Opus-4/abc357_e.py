# YOUR CODE HERE
def count_reachable_pairs(n, edges):
    total = 0
    
    for start in range(1, n + 1):
        visited = set()
        current = start
        
        # Follow edges until we revisit a vertex
        while current not in visited:
            visited.add(current)
            current = edges[current - 1]
        
        # Count of reachable vertices from start
        total += len(visited)
    
    return total

# Read input
n = int(input())
edges = list(map(int, input().split()))

# Solve and print result
result = count_reachable_pairs(n, edges)
print(result)