import collections # This import is not strictly needed for the final solution code, but harmless
from typing import List

class Solution:
    """
    Finds the maximum sum of squares of k elements achievable from nums
    after applying the bitwise AND/OR operation any number of times.
    The operation involves choosing two distinct indices i and j and simultaneously updating
    nums[i] to (nums[i] AND nums[j]) and nums[j] to (nums[i] OR nums[j]).
    """
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        :param nums: A 0-indexed integer array.
        :param k: A positive integer representing the number of elements to choose.
        :return: The maximum sum of squares of k chosen elements from the final array, modulo 10^9 + 7.
        """
        
        # Define the modulus
        MOD = 10**9 + 7
        
        # The core insight is that the operation (a, b) -> (a AND b, a OR b) preserves the sum a + b.
        # More importantly, it preserves the total count of set bits at each position p across the entire array.
        # Let N_p be the count of numbers in the initial `nums` array that have the p-th bit set.
        # Any sequence of operations will result in an array where the total count of set bits at position p remains N_p.
        # The operations effectively allow us to redistribute the set bits among the elements of the array, 
        # as long as the total count N_p for each bit position p is conserved.
        
        # To maximize the sum of squares of k chosen elements, we should aim to make these k elements as large as possible.
        # Since we can rearrange the bits, the optimal strategy is to concentrate the available set bits into k elements.
        # We can construct the k largest possible values greedily.
        
        # Calculate bit counts N_p for each bit position p.
        # The maximum value in nums is 10^9, which is less than 2^30.
        # So, we need to consider bits from position 0 up to 29. Iterating up to 30 (31 positions) is safe.
        bit_counts = [0] * 31 
        
        for num in nums:
            # For each number, iterate through its bits
            for p in range(31):
                # Check if the p-th bit is set
                if (num >> p) & 1:
                    bit_counts[p] += 1
                # Optimization note: We could potentially break early if (1 << p) > num,
                # as higher bits would be 0. But the current loop is simple and correct.
    
        # Construct the k target values greedily. Initialize k values to 0.
        # These values will represent the k largest numbers achievable.
        v = [0] * k
        
        # Iterate through each bit position p from 0 to 30.
        for p in range(31):
            # The value represented by the p-th bit is 2^p.
            pow2 = 1 << p
            
            # We have N_p = bit_counts[p] set bits available for position p.
            # To maximize the elements in `v`, assign these bits greedily starting from v[0].
            # We can assign this bit to at most N_p elements, and we only care about the top k elements.
            # So, assign the p-th bit to the first min(N_p, k) elements in `v`.
            count = min(bit_counts[p], k)
            
            # Add 2^p to the first 'count' elements in v.
            # This effectively sets the p-th bit for these elements.
            for i in range(count):
                v[i] += pow2
                
        # Calculate the sum of squares of the k constructed values, modulo MOD.
        total_sum_sq = 0
        for val in v:
            # Calculate the square of the value. Python handles large integers automatically.
            square_val = val * val 
            # Add the square (modulo MOD) to the total sum (modulo MOD).
            # Applying modulo at each addition step prevents the sum from potentially overflowing
            # if we were using fixed-size integers, and ensures the final result is correctly modded.
            total_sum_sq = (total_sum_sq + square_val % MOD) % MOD
            
        return total_sum_sq