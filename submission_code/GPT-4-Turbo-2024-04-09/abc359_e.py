def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:]))
    
    # This will hold the first time A_i > 0
    results = [0] * N
    
    # To find the first time A_i > 0, we need to simulate the process
    # We need to count the number of operations until A_i > 0 for each i
    current_op = 0
    A = [0] * (N + 1)
    
    # We need to keep track of when each A_i becomes positive
    first_positive = [float('inf')] * (N + 1)
    
    # Start the simulation
    i = 0
    while i < N:
        # Increase A_0 by 1
        A[0] += 1
        current_op += 1
        
        # Cascade the effect from A_0 to A_N
        for j in range(1, N + 1):
            if A[j - 1] > A[j] and A[j - 1] > H[j - 1]:
                A[j - 1] -= 1
                A[j] += 1
        
        # Check which A_i has become positive
        for j in range(1, N + 1):
            if A[j] > 0 and current_op < first_positive[j]:
                first_positive[j] = current_op
        
        # Check if all A_i have been positive at least once
        if all(first_positive[j] < float('inf') for j in range(1, N + 1)):
            break
        
        i += 1
    
    # Output the results
    print(" ".join(map(str, first_positive[1:])))