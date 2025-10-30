# YOUR CODE HERE
import sys
from collections import defaultdict

def find_strongest_programmer(N, M, relations):
    # Create a graph to represent the relations
    graph = defaultdict(list)

    # Add edges to the graph
    for A, B in relations:
        graph[A].append(B)

    # Function to perform DFS and find all reachable nodes
    def dfs(node, visited):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(graph[current])

    # Check if there is a unique strongest programmer
    for i in range(1, N + 1):
        visited = set()
        dfs(i, visited)
        if len(visited) == N - 1:
            return i

    return -1

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

relations = []
for i in range(M):
    A = int(data[2 * i + 2])
    B = int(data[2 * i + 3])
    relations.append((A, B))

# Find and print the strongest programmer
strongest = find_strongest_programmer(N, M, relations)
print(strongest)