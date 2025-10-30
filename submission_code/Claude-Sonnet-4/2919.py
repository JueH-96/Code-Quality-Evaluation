class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort in descending order to prioritize numbers with higher usage limits
        usageLimits.sort(reverse=True)
        
        groups = 0
        group_size = 1
        
        while True:
            # Check if we can form a group of current size
            # We need at least group_size numbers with positive usage limits
            available = 0
            for i in range(len(usageLimits)):
                if usageLimits[i] > 0:
                    available += 1
                else:
                    break
            
            if available < group_size:
                break
            
            # Form the group by using the first group_size numbers
            # (which have the highest remaining limits due to sorting)
            for i in range(group_size):
                usageLimits[i] -= 1
            
            # Re-sort to maintain descending order
            usageLimits.sort(reverse=True)
            
            groups += 1
            group_size += 1
        
        return groups