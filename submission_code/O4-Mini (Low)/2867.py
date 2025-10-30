class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # Collect indices of all 1's in the array
        one_positions = []
        for i, v in enumerate(nums):
            if v == 1:
                one_positions.append(i)
        
        # If there are no 1's, we can't form any good subarray
        if not one_positions:
            return 0
        
        # If there's exactly one 1, there's exactly one way:
        # the entire array is one good subarray
        if len(one_positions) == 1:
            return 1
        
        # For each gap between consecutive 1's of size g,
        # we have (g + 1) choices of where to place the cut.
        result = 1
        for j in range(1, len(one_positions)):
            gap = one_positions[j] - one_positions[j-1] - 1
            # multiply by (gap + 1)
            result = (result * (gap + 1)) % MOD
        
        return result