def find_good_sequence(N, M, constraints):
    from collections import defaultdict
    
    # Initialize the array A with None (unknown values)
    A = [None] * N
    
    # Create a graph to store the constraints
    graph = defaultdict(list)
    
    for x, y, z in constraints:
        x -= 1  # Convert to 0-based index
        y -= 1  # Convert to 0-based index
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    # To keep track of visited nodes
    visited = [False] * N
    
    def dfs(node, value):
        stack = [(node, value)]
        while stack:
            current, current_value = stack.pop()
            if A[current] is None:
                A[current] = current_value
            elif A[current] != current_value:
                return False
            
            visited[current] = True
            
            for neighbor, z in graph[current]:
                next_value = current_value ^ z
                if not visited[neighbor]:
                    stack.append((neighbor, next_value))
                elif A[neighbor] is not None and A[neighbor] != next_value:
                    return False
        return True
    
    # Iterate through all nodes to ensure all components are covered
    for i in range(N):
        if A[i] is None:
            if not dfs(i, 0):  # Start with 0 for the first unvisited node
                return -1
    
    # If we have a valid A, we need to minimize the sum
    # Replace None values with 0 to minimize the sum
    A = [0 if x is None else x for x in A]
    
    return A

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
constraints = [tuple(map(int, line.split())) for line in data[1:M+1]]

result = find_good_sequence(N, M, constraints)

if result == -1:
    print(-1)
else:
    print(' '.join(map(str, result)))