# YOUR CODE HERE
import sys

def solve():
    # Read integers N and M from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Define the modulus
    MOD = 998244353
    
    # Initialize the total sum
    total_sum = 0
    
    # The problem asks for the sum S = sum_{k=0}^{N} popcount(k & M) mod MOD.
    # popcount(x) is the number of set bits (1s) in the binary representation of x.
    # k & M denotes the bitwise AND operation.
    # The binary representation of x = k & M has x_i = 1 if and only if k_i = 1 and M_i = 1. Otherwise x_i = 0.
    # popcount(k & M) = sum_i (k_i AND M_i), where k_i and M_i are the i-th bits of k and M.
    # The total sum can be rewritten by swapping the order of summation:
    # S = sum_{k=0}^{N} sum_i (k_i AND M_i) = sum_i sum_{k=0}^{N} (k_i AND M_i)
    
    # The inner term (k_i AND M_i) depends on M_i:
    # If M_i = 0, then (k_i AND M_i) = 0 for all k. The contribution of bit i to the total sum is 0.
    # If M_i = 1, then (k_i AND M_i) = k_i. The contribution of bit i is sum_{k=0}^{N} k_i.
    
    # Let C_i = sum_{k=0}^{N} k_i. This is the count of integers k in the range [0, N] such that the i-th bit of k is 1.
    # The total sum S can be expressed as: S = sum_{i=0}^{59} [M_i == 1] * C_i mod MOD.
    # We need to consider bits up to 59 because N and M are less than 2^60.
    
    # Use N+1 for counting purposes as the range [0, N] contains N+1 integers.
    N_plus_1 = N + 1

    # Iterate through each bit position i from 0 up to 59.
    for i in range(60):
        # Check if the i-th bit of M is set (equal to 1).
        # We can do this using right shift and bitwise AND: (M >> i) & 1
        if (M >> i) & 1:
            # If M_i is 1, we need to calculate C_i and add it to the total sum.
            
            # Calculate powers of 2 related to bit position i.
            # pow2i = 2^i
            pow2i = 1 << i
            # pow2ip1 = 2^(i+1)
            pow2ip1 = 1 << (i + 1)
            
            # Calculate Ci = count of integers k in [0, N] such that k_i = 1.
            # We analyze the pattern of the i-th bit across integers. It cycles with period 2^(i+1):
            # 2^i zeros followed by 2^i ones.
            
            # First, count the number of full cycles of length 2^(i+1) within the first N+1 integers.
            # This is equivalent to floor((N+1) / 2^(i+1)).
            num_full_blocks = N_plus_1 // pow2ip1
            
            # Each full block contains exactly 2^i numbers where the i-th bit is 1.
            # Contribution from full blocks:
            term1 = num_full_blocks * pow2i
            
            # Next, consider the remaining integers after the full blocks.
            # The number of remaining integers is (N+1) % 2^(i+1).
            rem_len = N_plus_1 % pow2ip1
            
            # In this remaining range [0, rem_len-1], we count how many integers k have k_i = 1.
            # The integers with k_i = 1 within a block of size 2^(i+1) are those in the interval [2^i, 2^(i+1)-1].
            # We need to find the size of the intersection of [0, rem_len-1] and [2^i, 2^(i+1)-1].
            # This intersection count simplifies to max(0, rem_len - 2^i).
            term2 = max(0, rem_len - pow2i)
            
            # Ci is the sum of contributions from full blocks and the partial block.
            Ci = term1 + term2
            
            # Add the contribution Ci to the total sum.
            # Since Ci can be large (up to N+1 which can exceed MOD), we must take Ci modulo MOD before adding.
            total_sum = (total_sum + (Ci % MOD)) % MOD

    # Print the final computed sum.
    print(total_sum)

# Call the solve function to execute the logic.
solve()