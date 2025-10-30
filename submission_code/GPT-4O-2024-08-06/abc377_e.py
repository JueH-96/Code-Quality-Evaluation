# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    # Adjust P to be 0-indexed for easier handling
    P = [p - 1 for p in P]
    
    visited = [False] * N
    final_positions = [0] * N
    
    for start in range(N):
        if not visited[start]:
            # Find the cycle starting from 'start'
            cycle = []
            current = start
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current]
            
            cycle_length = len(cycle)
            # Calculate the effective number of operations needed
            effective_k = K % cycle_length
            
            # Determine the final position for each element in the cycle
            for i in range(cycle_length):
                final_positions[cycle[i]] = cycle[(i + effective_k) % cycle_length]
    
    # Convert back to 1-indexed and print the result
    result = [str(final_positions[i] + 1) for i in range(N)]
    print(" ".join(result))

if __name__ == "__main__":
    main()