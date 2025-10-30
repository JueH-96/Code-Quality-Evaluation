import collections
from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort usageLimits in ascending order.
        # This is a key step for the greedy strategy. By processing numbers with
        # smaller usage limits first, we ensure that we prioritize using our
        # "scarcer" resources efficiently.
        usageLimits.sort()

        # `formed_groups` tracks the total number of groups we have successfully formed.
        # These groups are implicitly assumed to have sizes 1, 2, ..., `formed_groups`.
        formed_groups = 0

        # `current_capacity_left` represents the accumulated "excess capacity"
        # from the numbers (with their respective usage limits) that we have processed so far.
        # More precisely, it holds the value of:
        # (sum of `usageLimits` of numbers processed) - (sum of sizes of groups already formed).
        # If this value is positive, it means we have "spare" usage capacity
        # that can contribute to forming future groups.
        current_capacity_left = 0

        # Iterate through each usage limit in the sorted list.
        for limit in usageLimits:
            # Add the current number's total available uses to our accumulated capacity.
            current_capacity_left += limit

            # Check if we have enough total capacity (`current_capacity_left`)
            # to form the next group. The next group would conceptually be of size
            # `formed_groups + 1` (since group sizes are 1, 2, 3, ...).
            # The condition `current_capacity_left >= formed_groups + 1`
            # essentially checks if `(sum of limits processed) - (sum of sizes of groups formed)`
            # is enough to cover `(next group size)`.
            # This simplifies to checking if `(sum of limits processed)`
            # is at least `(sum of all group sizes from 1 up to next group size)`.
            if current_capacity_left >= formed_groups + 1:
                # If the condition is true, it means we have sufficient resources
                # to form the next group.
                formed_groups += 1  # Increment the count of successfully formed groups.
                
                # After forming the new group, we must "pay" for the distinct items
                # required by it. The size of this newly formed group is now equal
                # to the updated `formed_groups` value.
                current_capacity_left -= formed_groups
            
            # If `current_capacity_left` is not enough, we cannot form the next group
            # with the items processed so far. We continue to the next `limit`, hoping
            # to accumulate more capacity.

        # After processing all usage limits, `formed_groups` holds the maximum
        # number of groups that can be formed satisfying all conditions.
        return formed_groups