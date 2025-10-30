class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Collect the indices of all 1's
        ones_positions = [i for i, val in enumerate(nums) if val == 1]
        k = len(ones_positions)
        
        # If no 1's, we cannot split into subarrays each having exactly one 1
        if k == 0:
            return 0
        # If there is exactly one 1, the entire array is one good subarray
        if k == 1:
            return 1
        
        # Otherwise, compute product of gaps between consecutive 1's
        ways = 1
        for i in range(k - 1):
            # Distance between consecutive 1's
            gap = ones_positions[i+1] - ones_positions[i]
            ways = (ways * gap) % MOD
        
        return ways % MOD