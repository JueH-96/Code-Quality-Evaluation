class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 1000000007
        
        # If no 1s in array, return 0 since we can't make any good subarrays
        if 1 not in nums:
            return 0
            
        # Find positions of all 1s
        one_positions = []
        for i in range(len(nums)):
            if nums[i] == 1:
                one_positions.append(i)
        
        # If only one 1, return 1 since only one possible split
        if len(one_positions) == 1:
            return 1
            
        # Calculate gaps between consecutive 1s
        result = 1
        for i in range(1, len(one_positions)):
            # Number of ways to split between two consecutive 1s is the gap size
            gap = one_positions[i] - one_positions[i-1]
            result = (result * gap) % MOD
            
        return result