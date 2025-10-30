class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        ones_positions = [i for i, value in enumerate(nums) if value == 1]
        
        if not ones_positions:
            return 0
        
        # Calculate the number of ways to split the array
        ways = 1
        for i in range(1, len(ones_positions)):
            distance = ones_positions[i] - ones_positions[i-1]
            ways *= distance
            ways %= 10**9 + 7
        
        return ways