class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * n
        suffix = [0] * n
        
        for i in range(1, n):
            prefix[i] = prefix[i-1] | nums[i-1]
        
        for i in range(n-2, -1, -1):
            suffix[i] = suffix[i+1] | nums[i+1]
        
        best = 0
        for i in range(n):
            candidate = prefix[i] | (nums[i] << k) | suffix[i]
            if candidate > best:
                best = candidate
        
        return best