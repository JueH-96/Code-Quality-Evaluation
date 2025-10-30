class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:

        def count_set_bits(num: int, i: int) -> int:
            """
            Counts the number of integers from 0 to num (inclusive)
            that have the i-th bit (1-indexed from right) set.
            """
            if i <= 0:
                return 0
            
            # The i-th bit (1-indexed) corresponds to 2^(i-1).
            # Block size = 2^i
            block_size = 1 << i
            # Half block size = 2^(i-1)
            half_block_size = 1 << (i - 1)
            
            num_plus_one = num + 1
            
            # Number of full blocks of size block_size in num_plus_one
            num_full_blocks = num_plus_one // block_size
            
            # Contribution from full blocks
            count = num_full_blocks * half_block_size
            
            # Remaining numbers in the partial block
            remaining_numbers = num_plus_one % block_size
            
            # Contribution from the partial block
            # The i-th bit is set for numbers from index half_block_size up to block_size - 1
            # within a block. In the range of remaining_numbers (0 to remaining_numbers-1),
            # we count how many indices fall into the range [half_block_size, block_size-1].
            # The count is max(0, remaining_numbers - half_block_size).
            count += max(0, remaining_numbers - half_block_size)
            
            return count

        def sum_prices(num: int, x: int) -> int:
            """
            Calculates the sum of prices for numbers from 1 to num.
            Price of n is the number of i's such that i % x == 0 and
            the i-th bit of n (1-indexed from right) is set.
            This is equal to the sum of CountSetBits(num, i) for all i >= 1 with i % x == 0.
            """
            total_price = 0
            
            # We need to consider bit positions i (1-indexed) such that i % x == 0.
            # The maximum relevant bit position is determined by the magnitude of num.
            # If num can be up to ~2^63 + 10^15, the highest possible set bit can be at position 64.
            # E.g., 2^63 has the 64th bit set. Numbers slightly larger might still have 64 as highest.
            # The maximum num we search for is around 2^63 + 10^15, which is less than 2^64.
            # The highest possible set bit for a number less than 2^64 is at position 64.
            # So we check relevant positions up to 1-indexed position 64.
            max_bit_pos_to_check = 64 # Check up to 1-indexed position 64. 2^63 uses bit 64.
            
            for i in range(x, max_bit_pos_to_check + 1, x):
                 # We need to count set bits at position i (1-indexed)
                 # for numbers 0 to num. Summing these counts gives the total price sum.
                 total_price += count_set_bits(num, i)
                 
            return total_price

        # Binary search for the greatest num such that sum_prices(num, x) <= k
        low = 0
        # Determine a sufficiently large upper bound for num.
        # As analyzed, the maximum possible answer can be around 2^63 + k - 1
        # when x is large (like 64) and k is large.
        # A safe upper bound should exceed this.
        # Let's use 2^63 + k + 100 as a safe upper bound.
        # (1 << 63) is 2^63.
        # Since k <= 10^15, (1 << 63) + k + 100 is roughly 9e18 + 1e15,
        # which fits comfortably in Python's arbitrary precision integers.
        high = (1 << 63) + k + 100 
        
        ans = 0

        while low <= high:
            mid = low + (high - low) // 2
            
            current_sum = sum_prices(mid, x)
            
            if current_sum <= k:
                # mid is a potential answer, try larger numbers
                ans = mid
                low = mid + 1
            else:
                # mid is too large, try smaller numbers
                high = mid - 1
        
        return ans