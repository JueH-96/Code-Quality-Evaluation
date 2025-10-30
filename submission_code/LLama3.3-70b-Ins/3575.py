from typing import List
import itertools

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        
        # Generate all possible subsequences of size 2 * k
        for subseq in itertools.combinations(range(n), 2 * k):
            # Split the subsequence into two halves
            first_half = [nums[i] for i in subseq[:k]]
            second_half = [nums[i] for i in subseq[k:]]
            
            # Calculate the value of the subsequence
            first_or = 0
            for num in first_half:
                first_or |= num
            second_or = 0
            for num in second_half:
                second_or |= num
            val = first_or ^ second_or
            
            # Update the maximum value
            max_val = max(max_val, val)
        
        return max_val