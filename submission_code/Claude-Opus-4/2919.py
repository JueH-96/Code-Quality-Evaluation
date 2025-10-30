class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort in descending order to prioritize numbers with higher limits
        usageLimits.sort(reverse=True)
        n = len(usageLimits)
        
        # Binary search on the number of groups
        left, right = 1, n
        result = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if we can form 'mid' groups
            if self.canFormGroups(usageLimits, mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def canFormGroups(self, usageLimits: List[int], k: int) -> bool:
        # We need groups of sizes 1, 2, 3, ..., k
        # Total elements needed: k(k+1)/2
        total_needed = k * (k + 1) // 2
        
        # Count how many elements we can use
        total_available = 0
        
        for i in range(len(usageLimits)):
            # Each number can be used in at most k groups (since we have k groups)
            # But it's also limited by its usage limit
            total_available += min(usageLimits[i], k)
            
            # Early termination if we have enough
            if total_available >= total_needed:
                return True
        
        return total_available >= total_needed