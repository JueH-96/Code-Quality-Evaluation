from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usageLimits array in non-decreasing order
        usageLimits.sort()

        # Initialize variables to keep track of the current group size and the total number of groups
        current_group_size = 0
        total_groups = 0

        # Iterate through the sorted usageLimits array
        for limit in usageLimits:
            # If the current group size is 0, start a new group
            if current_group_size == 0:
                current_group_size = 1
                total_groups += 1
            else:
                # If we can add the current number to the existing group, do so
                if limit >= current_group_size:
                    current_group_size += 1
                else:
                    # If we cannot add the current number to the existing group, start a new group
                    current_group_size = 1
                    total_groups += 1

        return total_groups