class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits in non-decreasing order
        usageLimits.sort()
        
        # This will keep track of the running sum of usage
        total = 0
        
        # This will keep track of how many groups we can form
        groups = 0
        
        # Iterate over the sorted usage limits
        for limit in usageLimits:
            # Add this limit to the running total
            total += limit

            # Check if we can form one more group (of size groups+1)
            # The total number of distinct "slots" needed for groups+1 groups is (groups+1)*(groups+2)//2
            if total >= (groups + 1) * (groups + 2) // 2:
                groups += 1
        
        return groups