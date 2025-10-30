import collections # collections is not used, but typically useful in Python competitive programming
from typing import List

class Solution:
    """
    Solves the numberOfGoodPartitions problem using logic based on finding minimal blocks.
    A partition of an array into contiguous subarrays is called "good" if no two subarrays contain the same number.
    This condition implies that for any number x present in the array, all its occurrences must lie within the same subarray of the partition.
    
    To satisfy this condition, if a number x first appears at index first[x] and last appears at index last[x],
    the entire range of indices [first[x], last[x]] must be contained within a single subarray.
    
    We can find the minimal contiguous blocks that must stay together by effectively merging overlapping intervals [first[x], last[x]].
    A more direct approach is to iterate through the array and track the furthest 'last occurrence' index (`max_reach`) for all numbers encountered so far in the current block.
    When the current index `i` equals `max_reach`, it marks the end of a minimal block that must stay together. Any partition cut can only happen between such blocks.
    
    If there are `k` such minimal blocks, the problem reduces to finding the number of ways to partition a sequence of `k` items.
    This is equivalent to deciding whether to place a cut or not in the `k-1` possible positions between adjacent blocks.
    There are 2 choices for each position (cut or no cut), leading to 2^(k-1) possible partitions.
    The result must be returned modulo 10^9 + 7.
    """
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        """
        Calculates the total number of good partitions of the array nums, modulo 10^9 + 7.

        Args:
            nums: A list of positive integers.

        Returns:
            The total number of good partitions modulo 10^9 + 7.
        """
        
        N = len(nums)
        # Base case: If the array has only one element, there's only one possible partition:
        # the array itself ([nums[0]]). This partition is always good.
        if N == 1:
            return 1
            
        # Store the last occurrence index for each distinct number in nums.
        # This helps determine the minimum extent of a subarray containing a number.
        last_pos = {}
        for i, x in enumerate(nums):
            last_pos[x] = i
            
        # `count` will store the number of minimal contiguous blocks identified.
        count = 0  
        # `max_reach` tracks the maximum rightmost index required by the current block being processed.
        # It ensures that all occurrences of any number within the block are indeed contained.
        # Initialized to 0, it will be updated based on the last occurrences of numbers encountered.
        max_reach = 0 
        
        # Iterate through the array to identify the minimal blocks.
        for i in range(N):
            # The current block must extend at least to the last occurrence index of nums[i].
            # Update max_reach to the maximum of its current value and the last position of nums[i].
            current_num_last_pos = last_pos[nums[i]]
            max_reach = max(max_reach, current_num_last_pos)
            
            # Check if the current index 'i' marks the end of a minimal block.
            # This occurs when 'i' equals `max_reach`. It implies that all numbers encountered
            # from the start of the current block up to index 'i' have their last occurrences
            # at or before index 'i'. No number starting in this block requires the block to extend further.
            # Therefore, a partition cut can be validly placed right after index 'i'.
            if i == max_reach:
                # Increment the count of minimal blocks found.
                count += 1
                # We don't need to explicitly reset max_reach for the next block; 
                # the loop naturally continues tracking the requirement for the next block starting from i+1.
        
        # The number of good partitions corresponds to the number of ways to partition
        # the sequence of 'count' minimal blocks identified. This is equivalent to placing cuts
        # in the 'count - 1' possible locations between adjacent blocks.
        # Each location offers 2 choices: either place a cut or not.
        # Thus, the total number of ways is 2^(count - 1).
        
        # Define the modulus value as required by the problem.
        MOD = 10**9 + 7
        
        # Calculate 2^(count - 1) modulo MOD using Python's built-in pow function.
        # pow(base, exponent, modulus) efficiently computes (base^exponent) % modulus.
        # It correctly handles the base case where count = 1: exponent becomes 0, and pow(2, 0, MOD) returns 1.
        result = pow(2, count - 1, MOD)
        
        return result