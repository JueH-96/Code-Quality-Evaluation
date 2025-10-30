class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all positions of the element 1 in nums
        ones_positions = [i for i, val in enumerate(nums) if val == 1]
        
        # If there are no 1s, we cannot form any "good" subarray partition
        if not ones_positions:
            return 0
        
        # If there's at least one 1, the number of ways to split between
        # consecutive 1s is the product of the gaps between them.
        ways = 1
        for i in range(len(ones_positions) - 1):
            ways = (ways * (ones_positions[i+1] - ones_positions[i])) % MOD
            
        return ways % MOD