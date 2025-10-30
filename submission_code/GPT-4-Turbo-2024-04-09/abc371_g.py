def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    
    # Convert P from 1-based index to 0-based index
    P = [p-1 for p in P]
    
    # Find cycles in the permutation P
    visited = [False] * N
    cycles = []
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            cycles.append(cycle)
    
    # For each cycle, find the lexicographically smallest arrangement
    result = [0] * N
    for cycle in cycles:
        cycle_values = [A[idx] for idx in cycle]
        cycle_values.sort()
        
        # Map sorted values back to positions in the cycle
        for i, idx in enumerate(cycle):
            result[idx] = cycle_values[i]
    
    # Print the result
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()