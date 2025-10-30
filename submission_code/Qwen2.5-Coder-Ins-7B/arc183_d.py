import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

for i in range(N-1):
    A = int(data[2*i+1])
    B = int(data[2*i+2])
    edges.append((A, B))

# Create a list of leaves
leaves = [i for i in range(1, N+1) if i not in [A for A, B in edges] and i not in [B for A, B in edges]]

# Perform the operations
for i in range(N//2):
    # Choose the two farthest leaves
    A = leaves[0]
    B = leaves[-1]
    print(A, B)
    
    # Remove the leaves from the list
    leaves.remove(A)
    leaves.remove(B)
    
    # Update the list of leaves
    leaves = [i for i in range(1, N+1) if i not in [A, B] and i not in [B for A, B in edges] and i not in [A for A, B in edges]]