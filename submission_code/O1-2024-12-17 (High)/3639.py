class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        coverage = [0] * (n + 1)  # Using a difference array approach
        
        # Mark the increments and decrements in the coverage array
        for l, r in queries:
            coverage[l] += 1
            if r + 1 < n:
                coverage[r + 1] -= 1
        
        # Build the actual coverage array via prefix sums
        for i in range(1, n):
            coverage[i] += coverage[i - 1]
        
        # Check if each index can be decremented enough times to reach zero
        for i in range(n):
            if nums[i] > coverage[i]:
                return False
        return True