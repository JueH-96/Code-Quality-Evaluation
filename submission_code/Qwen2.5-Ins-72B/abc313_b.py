# YOUR CODE HERE
import sys

def find_strongest_programmer(N, M, relations):
    # Initialize a graph to store the superiority relations
    graph = {i: set() for i in range(1, N + 1)}
    
    # Add the given relations to the graph
    for a, b in relations:
        graph[a].add(b)
    
    # Transitive closure to infer all possible relations
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if j in graph[i] or (k in graph[i] and j in graph[k]):
                    graph[i].add(j)
    
    # Check for the strongest programmer
    strongest = None
    for i in range(1, N + 1):
        if len(graph[i]) == N - 1:
            if strongest is not None:
                return -1  # Multiple possible strongest programmers
            strongest = i
    
    return strongest if strongest is not None else -1

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
relations = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(M)]

# Solve and print the result
print(find_strongest_programmer(N, M, relations))