import math
from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Max value of nums[i] is 10^9. 2^29 < 10^9 < 2^30.
        # So, we need to consider bits from position 0 up to 29 (inclusive),
        # which means a total of 30 bits.
        num_bits = 30 
        
        # bit_counts[p] will store the number of elements in the array
        # that have the p-th bit set. This count is an invariant property
        # of the array under the given operation.
        bit_counts = [0] * num_bits
        
        # Step 1: Count the occurrences of each bit across all numbers in the initial array.
        for num in nums:
            for p in range(num_bits):
                if (num >> p) & 1:  # Check if the p-th bit of 'num' is set
                    bit_counts[p] += 1
        
        # Step 2: Construct the 'k' numbers that will maximize the sum of their squares.
        # We achieve this by greedily assigning the available '1's for each bit position
        # to the 'k' numbers we are building, prioritizing higher value bits and larger numbers.
        
        # Initialize 'k' numbers to 0. These will eventually be the 'k' largest numbers
        # we can form from the initial array's bits.
        # ans_nums[0] will be the largest, ans_nums[1] the second largest, and so on.
        ans_nums = [0] * k 
        
        for p in range(num_bits):
            # For the current bit position 'p', bit_counts[p] tells us how many numbers
            # in the final array must have this bit set.
            # We want to assign this bit to the 'bit_counts[p]' largest numbers among our 'k' candidates.
            
            # Iterate through the 'k' numbers we are constructing.
            # 'i' represents the index of the number in ans_nums (0-indexed).
            # If bit_counts[p] > i, it means there are enough '1's at this bit position
            # for ans_nums[i] (which conceptually is the (i+1)-th largest number) to have this bit set.
            for i in range(k):
                if bit_counts[p] > i:
                    ans_nums[i] |= (1 << p) # Set the p-th bit for ans_nums[i]
                else:
                    # If bit_counts[p] <= i, it means we have exhausted all available '1's
                    # for this bit position, or we've assigned them to numbers larger than ans_nums[i].
                    # Since ans_nums are conceptually sorted in descending order of value,
                    # if ans_nums[i] doesn't get this bit, neither will ans_nums[i+1], ans_nums[i+2], etc.
                    # So, we can stop processing this bit position for the remaining numbers.
                    break 
        
        # Step 3: Calculate the sum of squares of the constructed 'k' numbers modulo MOD.
        total_sum_squares = 0
        for val in ans_nums:
            # Calculate val*val. It's important to take modulo at each step
            # to prevent intermediate results from exceeding integer limits,
            # especially since val can be up to 2^30 - 1 (approx 10^9), and val*val
            # can be up to 10^18, which exceeds standard 64-bit integer limits.
            # Python integers handle arbitrary precision, but explicit modulo
            # keeps the numbers within reasonable bounds for arithmetic speed
            # and follows typical competitive programming best practices.
            square_val = (val * val) % MOD
            total_sum_squares = (total_sum_squares + square_val) % MOD
            
        return total_sum_squares