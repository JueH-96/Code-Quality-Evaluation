class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Initialize a difference array to track the changes
        diff = [0] * (n + 1)
        
        # Apply the queries to the difference array
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r+1] -= 1
        
        # Compute the prefix sum to get the total decrements for each index
        total = 0
        for i in range(n):
            total += diff[i]
            if nums[i] < total:
                return False
        
        # Check if all elements are zero after applying all queries
        for i in range(n):
            if nums[i] != total:
                return False
            total -= diff[i]
        
        return True