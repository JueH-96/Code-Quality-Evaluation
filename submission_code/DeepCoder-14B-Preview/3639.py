class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        if n == 0:
            return True
        
        # Initialize the difference array
        diff = [0] * (n + 1)
        
        for l, r in queries:
            diff[l] += 1
            if r + 1 <= n:
                diff[r + 1] -= 1
        
        # Compute the prefix sum to determine K_i for each index
        current = 0
        for i in range(n):
            current += diff[i]
            if current < nums[i]:
                return False
        
        # Calculate total required decrements and possible decrements
        sum_total = sum(nums)
        sum_possible = sum((r - l + 1) for l, r in queries)
        
        if sum_total > sum_possible:
            return False
        
        return True