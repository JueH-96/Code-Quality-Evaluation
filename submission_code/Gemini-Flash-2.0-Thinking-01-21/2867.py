class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        indices = []
        for i in range(len(nums)):
            if nums[i] == 1:
                indices.append(i)
        
        if not indices:
            return 0
        if len(indices) == 1:
            return 1
        
        ans = 1
        mod = 10**9 + 7
        
        for i in range(1, len(indices)):
            zeros_count = indices[i] - indices[i-1] - 1
            ans = (ans * (zeros_count + 1)) % mod
            
        return ans