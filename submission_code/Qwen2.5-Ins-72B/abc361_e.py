import sys

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
for i in range(1, 2 * N - 1, 3):
    A = int(data[i])
    B = int(data[i + 1])
    C = int(data[i + 2])
    edges.append((A, B, C))

# Initialize the total distance
total_distance = 0

# Since the graph is a tree, the minimum distance to visit all cities is the sum of all edge lengths
for _, _, C in edges:
    total_distance += C

# The minimum travel distance required to visit all cities at least once
print(total_distance * 2 - max(C for _, _, C in edges))