class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all the positions of 1s in the array
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        
        # If there are no 1s, there are no good subarrays
        if not ones_positions:
            return 0
        
        # If there is only one 1, the whole array is the only good subarray
        if len(ones_positions) == 1:
            return 1
        
        # Calculate the number of ways to split the array into good subarrays
        ways = 1
        for i in range(1, len(ones_positions)):
            # The number of ways to split between two 1s is the number of zeros between them + 1
            ways *= (ones_positions[i] - ones_positions[i - 1])
            ways %= MOD
        
        return ways