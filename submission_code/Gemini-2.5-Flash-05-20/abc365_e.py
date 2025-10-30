import sys

def solve():
    N = int(sys.stdin.readline())
    # Read A as 0-indexed list for convenience.
    # The problem uses 1-indexed A_1, ..., A_N.
    # So A[0] is A_1, A[1] is A_2, ..., A[N-1] is A_N.
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix XORs
    # P[k] will store A_1 ^ A_2 ^ ... ^ A_k (using original 1-based indexing for A_i)
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 ^ A_2
    # ...
    # P[N] = A_1 ^ A_2 ^ ... ^ A_N
    P = [0] * (N + 1)
    current_xor_sum = 0
    for i in range(N):
        current_xor_sum ^= A[i] # A[i] is the (i+1)-th element from input
        P[i+1] = current_xor_sum

    # Max possible value for A_i is 10^8.
    # 2^26 = 67,108,864
    # 2^27 = 134,217,728
    # So, numbers can have bits up to index 26 (0-indexed).
    MAX_BITS = 27 

    total_sum_all_subarrays = 0

    # Step 1: Calculate sum of XORs for all contiguous subarrays (including length 1)
    # This sum covers S_{i,j} = P_j ^ P_{i-1} for all 1 <= i <= j <= N.
    # In terms of P array indices (0-indexed P_0, ..., P_N), this means summing P[y] ^ P[x]
    # for all 0 <= x < y <= N.
    for k in range(MAX_BITS):
        bit_k_contribution_all = 0
        
        # zeros_count_prefix: count of P[z] (where z < y) whose k-th bit is 0
        # ones_count_prefix: count of P[z] (where z < y) whose k-th bit is 1
        zeros_count_prefix = 0
        ones_count_prefix = 0

        # Iterate through P_0, P_1, ..., P_N using index 'y'
        for y in range(N + 1): 
            current_Py_k_bit = (P[y] >> k) & 1

            # When y=0 (processing P[0]), there are no x < 0, so no contribution from previous elements.
            # For y > 0, we sum with previous counts (P[0], ..., P[y-1]).
            # The k-th bit of (P[y] ^ P[x]) is 1 if current_Py_k_bit != k-th bit of P[x].
            if current_Py_k_bit == 0:
                bit_k_contribution_all += ones_count_prefix
            else: # current_Py_k_bit == 1
                bit_k_contribution_all += zeros_count_prefix
            
            # After using P[y] as the right endpoint (P_j), add P[y] to the counts for the next iteration.
            # These updated counts will represent P[0], ..., P[y] and be used when 'y+1' is the right endpoint.
            if current_Py_k_bit == 0:
                zeros_count_prefix += 1
            else:
                ones_count_prefix += 1
        
        total_sum_all_subarrays += bit_k_contribution_all * (1 << k)

    # Step 2: Subtract the sum of A_i values (subarrays of length 1)
    # The problem asks for sum for i < j, which explicitly excludes length 1 subarrays.
    # A_i is equivalent to P_i ^ P_{i-1}.
    sum_of_A = 0
    for i in range(1, N + 1): # Iterate from A_1 to A_N
        sum_of_A += (P[i] ^ P[i-1])

    final_answer = total_sum_all_subarrays - sum_of_A
    print(final_answer)

solve()