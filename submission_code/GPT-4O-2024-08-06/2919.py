class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits to try to use smaller limits first
        usageLimits.sort()
        
        # Initialize variables to track the number of groups and the current group size
        groups = 0
        current_group_size = 0
        
        # Iterate over the sorted usage limits
        for limit in usageLimits:
            # Add the current limit to the current group size
            current_group_size += limit
            
            # Check if we can form a new group with size `groups + 1`
            if current_group_size >= groups + 1:
                # Increment the number of groups
                groups += 1
                # Deduct the size of the new group from the current group size
                current_group_size -= groups
        
        return groups