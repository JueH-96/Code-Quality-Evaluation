# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict
    
    # Read edges
    edges = []
    degree = [0] * (N + 1)
    
    index = 2
    for _ in range(M):
        u = int(data[index])
        v = int(data[index + 1])
        edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        index += 2
    
    # Count vertices with odd degree
    odd_degree_vertices = [i for i in range(1, N + 1) if degree[i] % 2 == 1]
    
    # If the number of odd degree vertices is odd, it's impossible
    if len(odd_degree_vertices) % 2 == 1:
        print("No")
        return
    
    # If no edges, any assignment is valid
    if M == 0:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # If all degrees are even or exactly two odd degree vertices
    # Assign the same number to all vertices
    solution = [1] * N
    print("Yes")
    print(" ".join(map(str, solution)))