from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        Calculates the minimum number of operations required to make nums equal to target.
        An operation consists of selecting a subarray and incrementing or
        decrementing its elements by 1.

        The problem can be reframed as making the difference array
        diff[i] = target[i] - nums[i] all zeros using subarray
        increment/decrement operations on diff.
        An increment operation on nums[p..q] by 1 is equivalent to decrementing
        diff[p..q] by 1.
        A decrement operation on nums[p..q] by 1 is equivalent to incrementing
        diff[p..q] by 1.

        We can solve this by processing the difference array from left to right.
        Let d[i] = target[i] - nums[i].
        We track the required level of positive and negative changes needed from the left.
        active_positive_level: The total amount of net positive change required from the left
                                that is active at the start of the current index i.
        active_negative_level: The total amount of net negative change (absolute value)
                                required from the left that is active at the start of
                                the current index i.

        Args:
            nums: The input integer array.
            target: The target integer array.

        Returns:
            The minimum number of operations required.
        """
        n = len(nums)
        
        total_ops = 0
        
        # active_positive_level represents the total positive change required at index i-1
        # that must be carried over to index i.
        # active_negative_level represents the total negative change (abs value) required
        # at index i-1 that must be carried over to index i.
        # Initialize levels from the left (before index 0, the levels are 0)
        active_positive_level = 0
        active_negative_level = 0
        
        for i in range(n):
            # Calculate the required net change at the current index
            required_net_change = target[i] - nums[i]
            
            if required_net_change > 0:
                # The current required change is positive.
                # Any previously required negative change (active_negative_level) is not needed
                # for this positive requirement; it is effectively cancelled out.
                # The number of *new* positive operations that must start at this index is the
                # difference between the current positive requirement and the ongoing positive level.
                # The ongoing positive level (active_positive_level) represents the maximum positive
                # requirement encountered so far that is still "active".
                
                new_positive_ops = max(0, required_net_change - active_positive_level)
                total_ops += new_positive_ops
                
                # The new active positive level for the next index is the current required amount,
                # as this is the maximum positive level needed up to this point.
                active_positive_level = required_net_change
                
                # Since the net change at the current index is positive, there's no active
                # negative level required from this point onwards.
                active_negative_level = 0
                
            elif required_net_change < 0:
                # The current required change is negative.
                # Any previously required positive change (active_positive_level) is not needed
                # for this negative requirement; it is effectively cancelled out.
                # The number of *new* negative operations that must start at this index is the
                # difference between the absolute value of the current negative requirement
                # and the ongoing negative level (active_negative_level).
                
                new_negative_ops = max(0, abs(required_net_change) - active_negative_level)
                total_ops += new_negative_ops
                
                # The new active negative level (abs value) for the next index is the
                # absolute value of the current required amount, as this is the maximum
                # negative level (abs value) needed up to this point.
                active_negative_level = abs(required_net_change)
                
                # Since the net change at the current index is negative, there's no active
                # positive level required from this point onwards.
                active_positive_level = 0
                
            else: # required_net_change == 0
                # The current required change is zero.
                # This means the net effect needed is zero. Any active positive or negative
                # levels from the left are effectively cancelled out by each other or by
                # operations stopping exactly at i-1.
                # No net change is required from this point onwards.
                active_positive_level = 0
                active_negative_level = 0

        return total_ops