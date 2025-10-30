import sys

# Modulo constant
MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())

    total_sum = 0

    # Iterate through bit positions from 0 to 59.
    # N and M are up to 2^60 - 1, meaning they can have bits from 0 to 59.
    for i in range(60):
        # Check if the i-th bit of M is set.
        # If M's i-th bit is 0, then (k & M)'s i-th bit will always be 0,
        # regardless of k. So, this bit position contributes 0 to the sum.
        # If M's i-th bit is 1, then (k & M)'s i-th bit is 1 if and only
        # if k's i-th bit is 1.
        if (M >> i) & 1:
            # We need to count how many numbers k in the range [0, N]
            # have their i-th bit set.
            # This is equivalent to counting numbers in [0, (N+1)-1] that have the i-th bit set.
            
            # Calculate 2^i and 2^(i+1) for convenience.
            power_of_2_i = 1 << i          # Represents 2^i
            power_of_2_i_plus_1 = 1 << (i + 1) # Represents 2^(i+1)

            # We are counting numbers from 0 up to N (inclusive).
            # Let count_up_to be N + 1.
            count_up_to = N + 1

            # How many full blocks of size 2^(i+1) are there up to count_up_to?
            # Each full block contains 2^i numbers with the i-th bit set.
            num_full_blocks = count_up_to // power_of_2_i_plus_1
            
            # Contribution from all full blocks.
            # This can be a large number, so apply modulo.
            contribution_from_full_blocks = (num_full_blocks * power_of_2_i) % MOD

            # How many numbers in the remaining partial block (from the start of the block
            # to count_up_to - 1) have their i-th bit set?
            # The length of the partial block is count_up_to % power_of_2_i_plus_1.
            # These numbers effectively range from 0 to (count_up_to % power_of_2_i_plus_1) - 1
            # relative to the start of their block.
            # The numbers in this relative range that have their i-th bit set are
            # those in [2^i, 2^(i+1)-1].
            # So, we are looking for numbers x such that 2^i <= x < remaining_length_in_partial_block.
            # The count is max(0, remaining_length_in_partial_block - 2^i).
            
            remaining_length_in_partial_block = count_up_to % power_of_2_i_plus_1
            
            # This gives the count of numbers in [0, remaining_length_in_partial_block - 1]
            # that have their i-th bit set.
            contribution_from_partial_block = max(0, remaining_length_in_partial_block - power_of_2_i)
            
            # Total count of k in [0, N] with i-th bit set.
            count_for_this_bit_pos = (contribution_from_full_blocks + contribution_from_partial_block) % MOD
            
            # Add this count to the total sum.
            total_sum = (total_sum + count_for_this_bit_pos) % MOD

    # Print the final result modulo 998244353.
    print(total_sum)

# Call the solve function to execute the program
solve()