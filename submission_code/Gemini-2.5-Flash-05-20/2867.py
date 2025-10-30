from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Step 1: Find all indices of '1's in the array.
        ones_indices = []
        for i, num in enumerate(nums):
            if num == 1:
                ones_indices.append(i)
        
        # Step 2: Handle edge cases based on the count of '1's.
        
        # Case 1: If there are no '1's in the array.
        # A "good" subarray must contain exactly one '1'.
        # If there are no '1's, it's impossible to form any good subarray,
        # so there are 0 ways to split the array.
        if not ones_indices:
            return 0
        
        # Case 2: If there is exactly one '1' in the array.
        # The entire array itself must form a single "good" subarray.
        # There is only 1 way to achieve this split (i.e., not splitting it further).
        if len(ones_indices) == 1:
            return 1
            
        # Step 3: If there are multiple '1's in the array.
        # Each '1' must belong to a unique good subarray. This means if there are K '1's,
        # the array must be split into K good subarrays.
        # The crucial observation is that the split points must occur in the segments of '0's
        # between consecutive '1's.
        #
        # For any two consecutive '1's at indices `ones_indices[i]` and `ones_indices[i+1]`,
        # let `p` be the index where the current good subarray ends and the next good subarray begins.
        # The current '1' (`nums[ones_indices[i]]`) must be part of the current subarray,
        # meaning `ones_indices[i] <= p`.
        # The next '1' (`nums[ones_indices[i+1]]`) must be part of the next subarray,
        # meaning `p + 1 <= ones_indices[i+1]`, which implies `p < ones_indices[i+1]`.
        #
        # So, the valid range for the split point `p` is `[ones_indices[i], ones_indices[i+1] - 1]`.
        # The number of choices for `p` in this range is `(ones_indices[i+1] - 1) - ones_indices[i] + 1`,
        # which simplifies to `ones_indices[i+1] - ones_indices[i]`.
        # This value represents the number of zeros between the two '1's plus one (for the split right after the first '1').
        #
        # The total number of ways to split the array is the product of the number of choices
        # for each intermediate split point between consecutive '1's.
        
        result = 1
        # Iterate through consecutive pairs of '1's indices to find the number of ways to split between them.
        for i in range(len(ones_indices) - 1):
            # Calculate the number of possible positions for the split between ones_indices[i] and ones_indices[i+1]
            diff = ones_indices[i+1] - ones_indices[i]
            
            # Multiply the result by the number of choices for this split, taking modulo
            result = (result * diff) % MOD
            
        return result