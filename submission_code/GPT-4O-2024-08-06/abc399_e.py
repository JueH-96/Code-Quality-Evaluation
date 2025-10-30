# YOUR CODE HERE
def min_operations_to_transform(N, S, T):
    if S == T:
        return 0
    
    # Create a mapping from S to T
    mapping = {}
    for s, t in zip(S, T):
        if s in mapping:
            if mapping[s] != t:
                return -1  # Conflicting mappings
        else:
            mapping[s] = t
    
    # Check for cycles
    visited = {}
    def has_cycle(node, path):
        if node in path:
            return True
        if node in visited:
            return False
        visited[node] = True
        path.add(node)
        if node in mapping:
            if has_cycle(mapping[node], path):
                return True
        path.remove(node)
        return False
    
    # Check for cycles in the graph
    for node in mapping:
        if has_cycle(node, set()):
            return -1
    
    # Count the number of unique transformations
    unique_transformations = set(mapping.values())
    return len(unique_transformations)

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
T = data[2]

# Output the result
print(min_operations_to_transform(N, S, T))