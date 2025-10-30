class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] | nums[i - 1]
        
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        max_or = prefix[n]  # Original OR of all elements
        
        for i in range(n):
            shifted = nums[i] << k
            current_or = prefix[i] | shifted | suffix[i + 1]
            if current_or > max_or:
                max_or = current_or
        
        return max_or