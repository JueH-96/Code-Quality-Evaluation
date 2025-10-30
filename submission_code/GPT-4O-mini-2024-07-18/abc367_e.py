def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:N+2]))
    A = list(map(int, data[N+2:2*N+2]))
    
    # Convert X from 1-based to 0-based
    X = [x - 1 for x in X]
    
    # To find the final state of A after K operations
    # We will use a cycle detection approach
    seen = {}
    current = A[:]
    
    for step in range(K):
        # Create a new array B based on current A and X
        next_A = [current[X[i]] for i in range(N)]
        
        # Check if we have seen this configuration before
        state_tuple = tuple(next_A)
        if state_tuple in seen:
            # Cycle detected
            cycle_length = step + 1 - seen[state_tuple]
            remaining_steps = (K - (step + 1)) % cycle_length
            # We need to perform remaining_steps more transformations
            for _ in range(remaining_steps):
                next_A = [next_A[X[i]] for i in range(N)]
            print(" ".join(map(str, next_A)))
            return
        else:
            seen[state_tuple] = step + 1
        
        current = next_A
    
    print(" ".join(map(str, current)))

if __name__ == "__main__":
    main()