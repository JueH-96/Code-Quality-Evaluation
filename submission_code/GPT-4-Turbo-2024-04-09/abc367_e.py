def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:N+2]))
    A = list(map(int, data[N+2:2*N+2]))
    
    # Convert X to 0-based index
    X = [x - 1 for x in X]
    
    # To handle large K efficiently, we need to detect cycles in the permutation
    # Apply the permutation until we find a cycle or reach K times
    visited = [False] * N
    cycles = []
    
    # Find all cycles in the permutation
    for i in range(N):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = X[current]
            cycles.append(cycle)
    
    # Result array
    result = [0] * N
    
    # Process each cycle
    for cycle in cycles:
        cycle_length = len(cycle)
        # Effective K to consider after modulo with cycle length
        effective_k = K % cycle_length
        
        # Rearrange elements in the cycle according to K
        for idx in range(cycle_length):
            new_position = cycle[(idx + effective_k) % cycle_length]
            result[new_position] = A[cycle[idx]]
    
    # Print the result
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()