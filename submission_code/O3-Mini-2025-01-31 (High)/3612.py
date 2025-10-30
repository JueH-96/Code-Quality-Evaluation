from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        # Define a helper function to check whether a subarray is strictly increasing.
        def is_strictly_increasing(sub):
            # A subarray of length 0 or 1 is trivially strictly increasing.
            return all(sub[i] < sub[i+1] for i in range(len(sub)-1))
        
        n = len(nums)
        # Iterate over all possible starting indices for the first subarray.
        # We need to ensure there are 2*k numbers into nums starting from the index.
        for i in range(n - 2*k + 1):
            first = nums[i:i+k]
            second = nums[i+k:i+2*k]
            if is_strictly_increasing(first) and is_strictly_increasing(second):
                return True
        return False