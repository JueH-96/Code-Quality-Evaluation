class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
        
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        max_or = 0
        for i in range(n):
            current = prefix[i] | (nums[i] << k) | suffix[i + 1]
            max_or = max(max_or, current)
        
        return max_or