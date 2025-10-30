class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        mod = 10**9 + 7
        
        if not ones:
            return 0
        if len(ones) == 1:
            return 1
        
        result = 1
        for i in range(len(ones) - 1):
            result = (result * (ones[i+1] - ones[i])) % mod
        
        return result