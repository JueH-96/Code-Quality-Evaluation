class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        coverage = [0] * (n + 1)  # Using n+1 to handle boundary calculations
        
        # Use prefix sum technique to efficiently calculate query coverage
        for l, r in queries:
            coverage[l] += 1        # Mark the start of the query range
            coverage[r + 1] -= 1    # Mark the end of the query range
        
        # Calculate prefix sum to get total coverage at each position
        for i in range(1, n + 1):
            coverage[i] += coverage[i - 1]
        
        # Check if each position has enough coverage to reach zero
        for i in range(n):
            if coverage[i] < nums[i]:
                return False
        
        return True