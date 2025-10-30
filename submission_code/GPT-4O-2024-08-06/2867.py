class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all the indices where the value is 1
        ones_indices = [i for i, num in enumerate(nums) if num == 1]
        
        # If there are no 1s, there are no good subarrays
        if not ones_indices:
            return 0
        
        # If there is only one 1, the whole array is the only good subarray
        if len(ones_indices) == 1:
            return 1
        
        # Calculate the number of ways to split the array into good subarrays
        # by counting the number of zeros between consecutive 1s
        ways = 1
        for i in range(1, len(ones_indices)):
            # The number of zeros between two consecutive 1s
            zeros_between = ones_indices[i] - ones_indices[i - 1] - 1
            # Each zero can be a split point, so we have (zeros_between + 1) choices
            ways *= (zeros_between + 1)
            ways %= MOD
        
        return ways