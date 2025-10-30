def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:2+N]))
    A = list(map(int, data[2+N:2+2*N]))
    
    # Precompute the mapping for each step
    # We need to find the K-th power of the permutation X
    # Since K can be up to 1e18, we need to find the cycle for each element
    
    # First, find the cycles in the permutation X
    visited = [False] * (N + 1)
    cycles = []
    
    for i in range(1, N+1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = X[current-1]
            cycles.append(cycle)
    
    # For each element, find its position in its cycle and compute the K-th step
    # We need to map each element to its final position after K steps
    # So, for each element, we find its cycle and its position in the cycle
    # Then, the final position is (pos + K) % len(cycle)
    
    # Create a dictionary to map each element to its cycle and position
    element_to_cycle = {}
    for cycle in cycles:
        for idx, elem in enumerate(cycle):
            element_to_cycle[elem] = (cycle, idx)
    
    # Now, for each element in A, we find its final position after K steps
    # The final A_i is A_{X_i^{K}}
    # So, we need to find X_i^{K}
    
    # To find X_i^{K}, we find the cycle of X_i and compute the K-th step in the cycle
    # So, for each i, we find the cycle of X_i and its position in the cycle
    # Then, the final position is (pos + K) % len(cycle)
    
    # So, for each i, we find the cycle of X_i and its position in the cycle
    # Then, the final position is (pos + K) % len(cycle)
    # Then, the final X_i^{K} is cycle[(pos + K) % len(cycle)]
    
    # So, for each i, we compute X_i^{K} and then A_i = A_{X_i^{K}}
    
    # Precompute the final X_i^{K} for each i
    final_X = [0] * N
    for i in range(N):
        x_i = X[i]
        cycle, pos = element_to_cycle[x_i]
        final_pos = (pos + K) % len(cycle)
        final_X[i] = cycle[final_pos]
    
    # Now, compute the final A
    final_A = [A[final_X[i] - 1] for i in range(N)]
    
    # Print the final A
    print(' '.join(map(str, final_A)))

if __name__ == "__main__":
    main()