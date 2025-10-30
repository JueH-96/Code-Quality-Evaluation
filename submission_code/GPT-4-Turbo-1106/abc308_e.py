def calculate_mex_sum(N, A, S):
    # Initialize counters for each character in S
    M_count = E_count = X_count = 0
    # Initialize prefix sums for the counts of 0s, 1s, and 2s in A
    prefix_0 = [0] * (N + 1)
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    
    # Calculate prefix sums
    for i in range(N):
        prefix_0[i + 1] = prefix_0[i] + (A[i] == 0)
        prefix_1[i + 1] = prefix_1[i] + (A[i] == 1)
        prefix_2[i + 1] = prefix_2[i] + (A[i] == 2)
    
    # Initialize the result
    result = 0
    
    # Iterate over all possible j (middle index)
    for j in range(1, N - 1):
        # Update the character counts based on S[j]
        if S[j] == 'M':
            M_count += 1
        elif S[j] == 'E':
            E_count += 1
        elif S[j] == 'X':
            X_count += 1
        
        # If the current character is 'E', calculate the contribution to the result
        if S[j] == 'E':
            # Count the number of 0s, 1s, and 2s before and after the current position j
            count_0_before = prefix_0[j]
            count_1_before = prefix_1[j] - prefix_0[j]
            count_2_before = prefix_2[j] - prefix_1[j]
            count_0_after = prefix_0[N] - prefix_0[j + 1]
            count_1_after = prefix_1[N] - prefix_1[j + 1] - count_0_after
            count_2_after = prefix_2[N] - prefix_2[j + 1] - count_1_after - count_0_after
            
            # Calculate the contribution to the result based on the counts of M and X before and after j
            result += M_count * count_0_after + X_count * count_1_after
            result += (M_count - count_0_before) * count_2_after + (X_count - count_1_before - count_0_before) * count_0_after
    
    # Output the result
    print(result)

# Read input
N = int(input().strip())
A = list(map(int, input().strip().split()))
S = input().strip()

# Solve the problem
calculate_mex_sum(N, A, S)