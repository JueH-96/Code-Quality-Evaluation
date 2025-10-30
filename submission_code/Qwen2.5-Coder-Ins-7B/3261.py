class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]
        
        suffix_and = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_and[i] = suffix_and[i + 1] & nums[i]
        
        min_or = float('inf')
        for i in range(n - k):
            min_or = min(min_or, prefix_or[i] | suffix_and[i + k + 1])
        
        return min_or