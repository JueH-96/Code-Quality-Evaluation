def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:N+2]))
    
    # Convert P to zero-indexed for easier manipulation
    P = [p - 1 for p in P]
    
    # To find the final permutation after K operations, we can use the concept of cycles
    visited = [False] * N
    cycles = []
    
    # Find all cycles in the permutation
    for i in range(N):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            cycles.append(cycle)
    
    # Now we need to apply the K operations to each cycle
    result = [0] * N
    for cycle in cycles:
        cycle_length = len(cycle)
        # Effective number of operations we need to perform
        effective_k = K % cycle_length
        
        # Update the result based on the effective operations
        for j in range(cycle_length):
            result[cycle[j]] = P[cycle[(j + effective_k) % cycle_length]]
    
    # Convert result back to 1-indexed for output
    result = [x + 1 for x in result]
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()