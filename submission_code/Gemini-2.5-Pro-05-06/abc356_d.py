import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    ans = 0
    # Iterate through bit positions 0 to 59.
    # N, M are at most 2^60 - 1, so they fit within 60 bits (indices 0 to 59).
    for i in range(60): # bit index i
        # If i-th bit of M is 0, then (k & M)'s i-th bit is also 0 for any k.
        # So this bit position i contributes nothing to popcount(k & M).
        if not ((M >> i) & 1):
            continue

        # If i-th bit of M is 1, then (k & M)'s i-th bit is 1 if and only if k's i-th bit is 1.
        # So, for this bit position i, we need to count how many numbers k in [0, N]
        # have their i-th bit set. Let this count be C_i_N.
        # Each such k will contribute 1 to the total sum of popcounts, via this specific bit i.
        
        # Calculation of C_i_N:
        # Count of numbers k in [0, N] such that k's i-th bit is 1.
        
        # The numbers k range from 0 to N, so there are N+1 values for k.
        num_values_k = N + 1
        
        power_of_2_i = 1 << i
        # Blocks are of size 2 * power_of_2_i = 1 << (i+1)
        block_size = 1 << (i + 1) 
        
        # Number of full blocks of size 'block_size'
        num_full_blocks = num_values_k // block_size
        
        # Each full block contains 'power_of_2_i' numbers where the i-th bit is set.
        count_from_full_blocks = num_full_blocks * power_of_2_i
        
        # Remaining part (partial block)
        remainder_len = num_values_k % block_size
        
        # In this partial block of length 'remainder_len' (representing values from 0 to remainder_len-1
        # relative to the start of a conceptual block), numbers with i-th bit set are those
        # from power_of_2_i up to remainder_len-1.
        # The count is max(0, remainder_len - power_of_2_i).
        count_from_partial_block = 0
        # This check is equivalent to max(0, remainder_len - power_of_2_i)
        if remainder_len > power_of_2_i: 
            count_from_partial_block = remainder_len - power_of_2_i
        
        C_i_N = count_from_full_blocks + count_from_partial_block
        
        # Add this count (modulo MOD) to the total answer.
        # C_i_N itself can be very large, Python handles large integers.
        # The sum (ans + C_i_N) is computed first, then modulo is applied.
        ans = (ans + C_i_N) % MOD
            
    sys.stdout.write(str(ans) + "
")

solve()