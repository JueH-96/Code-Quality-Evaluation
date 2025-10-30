class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Difference array to count coverage
        diff = [0] * (n + 1)
        
        # Build the difference array
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
                
        # Calculate coverage using prefix sums
        coverage = [0] * n
        coverage[0] = diff[0]
        for i in range(1, n):
            coverage[i] = coverage[i - 1] + diff[i]
            
        # Check if, for each index i, coverage[i] >= nums[i]
        for i in range(n):
            if coverage[i] < nums[i]:
                return False
        
        return True