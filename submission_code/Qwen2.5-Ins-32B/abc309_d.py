import sys

def find_max_d(n1, n2, m, edges):
    # Initialize sets to keep track of connected components
    connected_to_1 = {1}
    connected_to_n1_n2 = {n1 + n2}
    
    # Process edges to find connected components
    for a, b in edges:
        if a <= n1 and b > n1:
            connected_to_1.add(a)
            connected_to_n1_n2.add(b)
        elif b <= n1 and a > n1:
            connected_to_1.add(b)
            connected_to_n1_n2.add(a)
    
    # Calculate the maximum possible d
    max_d = 2 + max(len(connected_to_1), len(connected_to_n1_n2))
    
    return max_d

# Read input
n1, n2, m = map(int, input().split())
edges = [tuple(map(int, line.split())) for line in sys.stdin.readlines()]

# Solve the problem
max_d = find_max_d(n1, n2, m, edges)

# Print the answer
print(max_d)