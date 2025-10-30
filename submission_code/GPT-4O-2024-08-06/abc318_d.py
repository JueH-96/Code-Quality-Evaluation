# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    edges = []
    
    # Read the upper triangular matrix of weights
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            weight = int(data[index])
            index += 1
            edges.append((weight, i, j))
    
    # Sort edges by weight in descending order
    edges.sort(reverse=True, key=lambda x: x[0])
    
    used_vertices = set()
    max_weight = 0
    
    # Select edges greedily
    for weight, u, v in edges:
        if u not in used_vertices and v not in used_vertices:
            max_weight += weight
            used_vertices.add(u)
            used_vertices.add(v)
    
    print(max_weight)

if __name__ == "__main__":
    main()