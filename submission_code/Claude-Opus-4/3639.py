class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        
        # Create a difference array to track query coverage
        diff = [0] * (n + 1)
        
        # Mark the start and end of each query range
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        # Calculate the actual coverage at each position
        coverage = 0
        for i in range(n):
            coverage += diff[i]
            # If any position has more value than available queries, return False
            if nums[i] > coverage:
                return False
        
        return True