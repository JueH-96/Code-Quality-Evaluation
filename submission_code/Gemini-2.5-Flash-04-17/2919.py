from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        """
        Calculates the maximum number of groups with strictly increasing sizes
        (1, 2, 3, ...) that can be formed using items with given usage limits.

        Args:
            usageLimits: A list of integers where usageLimits[i] is the
                         maximum number of times item i can be used in total
                         across all groups. Items are numbered from 0 to
                         len(usageLimits) - 1.

        Returns:
            The maximum number of groups that can be formed while satisfying
            the conditions: each group has distinct numbers, and each group
            (except the first) is strictly larger than the previous one.
        """
        # Sort usage limits in ascending order.
        # This greedy approach processes items with smaller usage limits first.
        # The intuition is that items with fewer uses are more restrictive resources,
        # so we should leverage their limited capacity early for smaller group demands.
        # This maximizes the potential for the overall number of groups by leaving
        # the more abundant items for the larger groups that require more items.
        usageLimits.sort()

        # available_capacity keeps track of the total number of item uses from
        # all processed item types (sum of their sorted limits) that have not
        # yet been assigned to fulfill the requirements of the groups formed
        # so far. It represents a pool of "credits" that can be used to satisfy
        # the demand for distinct items in future groups.
        # Initially, we have no items and no groups.
        available_capacity = 0

        # groups_count tracks the number of groups we have successfully committed
        # to forming. The groups will have sizes 1, 2, 3, ..., groups_count.
        # We start attempting to form the first group (size 1).
        groups_count = 0

        # Iterate through the sorted usage limits, processing item types one by one.
        for limit in usageLimits:
            # Add the current item's usage limit to our available capacity.
            # This item contributes 'limit' potential uses across different groups
            # over the course of the whole process.
            available_capacity += limit

            # Check if we have enough accumulated capacity to form the next group.
            # The next group required is group number (groups_count + 1), which needs
            # groups_count + 1 distinct items.
            # The condition available_capacity >= groups_count + 1 checks if the total
            # accumulated capacity from the items processed so far (available_capacity)
            # is sufficient to conceptually provide the distinct items needed for the
            # next group of size groups_count + 1. If this condition is met, it means
            # that even using the most constrained items first, we still have enough
            # total flexibility to add another group of the next increasing size.
            if available_capacity >= groups_count + 1:
                # We can form the group with size groups_count + 1.
                # Forming this group requires groups_count + 1 conceptual item slots
                # from our total available pool. We update available_capacity by
                # subtracting this amount. The remaining available_capacity carries
                # over to help form subsequent, larger groups.
                available_capacity -= (groups_count + 1)

                # We have successfully formed one more group.
                groups_count += 1

        # After iterating through all item types, groups_count holds the maximum
        # number of increasing groups (sizes 1, 2, ..., groups_count) that could
        # be formed respecting the usage limits and distinctness constraint.
        return groups_count