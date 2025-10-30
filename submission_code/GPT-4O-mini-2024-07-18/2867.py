class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = 0
        last_one_index = -1
        ways = 1
        
        for i in range(len(nums)):
            if nums[i] == 1:
                if last_one_index != -1:
                    # Calculate the number of zeros between the last 1 and the current 1
                    zeros_between = i - last_one_index - 1
                    ways = (ways * (zeros_between + 1)) % MOD
                last_one_index = i
        
        return ways if last_one_index != -1 else 0