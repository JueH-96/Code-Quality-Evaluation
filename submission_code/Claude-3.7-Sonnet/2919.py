class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort is optional but makes the solution more intuitive
        usageLimits.sort()
        
        left, right = 1, len(usageLimits)
        
        while left <= right:
            mid = (left + right) // 2
            
            # Calculate total available slots when forming 'mid' groups
            total_slots = sum(min(usage, mid) for usage in usageLimits)
            
            # Check if we have enough slots to form 'mid' groups
            if total_slots >= mid * (mid + 1) // 2:
                left = mid + 1
            else:
                right = mid - 1
        
        return right