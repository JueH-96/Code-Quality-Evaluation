def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Convert A to 0-based indexing
    graph = [a-1 for a in A]
    
    # Initialize variables
    path = []
    visited = set()
    
    # Start from vertex 1 (index 0)
    current = 0
    while current not in visited:
        path.append(current)
        visited.add(current)
        current = graph[current]
    
    # Find the start of the cycle
    start = path.index(current)
    cycle = path[start:] + [path[start]]  # Include the starting vertex again to complete the cycle
    M = len(cycle)
    
    # Convert back to 1-based indexing for output
    cycle_str = ' '.join(str(x+1) for x in cycle[:-1])  # Exclude the last duplicate start vertex
    print(M-1)  # M is the number of vertices in the cycle
    print(cycle_str)

if __name__ == '__main__':
    main()