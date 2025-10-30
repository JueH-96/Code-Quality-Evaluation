class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones_count = nums.count(1)
        
        if ones_count == 0:
            return 0
        
        # Prefix sum to count number of ones until the current position
        prefix_ones = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_ones[i + 1] = prefix_ones[i] + nums[i]
        
        ways = 0
        for i in range(1, len(nums)):
            if prefix_ones[i] == 1 and prefix_ones[-1] - prefix_ones[i] == ones_count - 1:
                ways += 1
        
        return ways % MOD