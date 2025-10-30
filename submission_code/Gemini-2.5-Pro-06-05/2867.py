from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 1_000_000_007

        # Find all indices where the element is 1.
        one_indices = [i for i, num in enumerate(nums) if num == 1]

        # Case 1: No 1s in the array.
        # A "good subarray" must contain exactly one 1.
        # If there are no 1s, it's impossible to form any good subarray.
        # Therefore, there are 0 ways to split the array.
        if not one_indices:
            return 0
        
        # Case 2: There is at least one 1.
        # The number of ways to split the array is determined by the number of
        # possible cut points between consecutive 1s.
        #
        # Let the indices of two consecutive 1s be p_i and p_{i+1}.
        # The first subarray of this pair must contain the 1 at p_i, and the
        # second must contain the 1 at p_{i+1}. The split point can be placed
        # after any of the elements from index p_i to p_{i+1}-1.
        # The number of choices for this split point is (p_{i+1}-1) - p_i + 1 = p_{i+1} - p_i.
        #
        # The total number of ways to split the entire array is the product of
        # these choices for all consecutive pairs of 1s.
        
        # We start with 1 because it's the identity for multiplication.
        # If there's only one 1, the loop won't run, and we correctly return 1.
        result = 1
        for i in range(len(one_indices) - 1):
            diff = one_indices[i+1] - one_indices[i]
            result = (result * diff) % MOD
            
        return result