# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:N+2]))
    A = list(map(int, data[N+2:2*N+2]))
    
    # Adjust X to be zero-indexed
    X = [x - 1 for x in X]
    
    if K == 0:
        # If K is 0, no operations are performed, output A as is
        print(' '.join(map(str, A)))
        return
    
    # To handle the large K, we need to find the cycle in the permutation
    visited = [False] * N
    cycle_length = 0
    current_A = A[:]
    
    # Find the cycle length
    for start in range(N):
        if not visited[start]:
            cycle = []
            i = start
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                i = X[i]
            cycle_length = max(cycle_length, len(cycle))
    
    # Effective number of operations is K % cycle_length
    effective_K = K % cycle_length
    
    # Perform the effective number of operations
    for _ in range(effective_K):
        next_A = [0] * N
        for i in range(N):
            next_A[i] = current_A[X[i]]
        current_A = next_A
    
    print(' '.join(map(str, current_A)))

if __name__ == "__main__":
    main()