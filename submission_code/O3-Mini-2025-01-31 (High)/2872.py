from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # The idea is to merge segments from right to left.
        # We can merge two adjacent segments if the left segment's value (or merged sum) is
        # less than or equal to the right segment's value. In reverse order, if we let
        # "curr" be the sum of a segment (starting with the last element), then for each
        # element nums[i] (scanning backwards) if nums[i] <= curr, it means we can merge
        # nums[i] with the segment represented by curr. Otherwise, we cannot merge, so we
        # finalize the current segment and start a new segment with nums[i].
        #
        # Finally, the answer is the maximum segment sum formed in this way.
        
        n = len(nums)
        curr = nums[-1]  # current merged segment sum from the right end
        ans = curr
        
        # Process the array in reverse order
        for i in range(n - 2, -1, -1):
            # If the left value is not greater than the current segment sum,
            # they can be merged.
            if nums[i] <= curr:
                curr += nums[i]
            else:
                # Otherwise, finalize the current segment and update the maximum.
                ans = max(ans, curr)
                curr = nums[i]
        # Check for the last segment we were building.
        return max(ans, curr)


# The following section is for local testing.
if __name__ == "__main__":
    sol = Solution()
    # Example 1: Expected output is 21.
    print(sol.maxArrayValue([2, 3, 7, 9, 3]))  # Output: 21
    
    # Example 2: Expected output is 11.
    print(sol.maxArrayValue([5, 3, 3]))  # Output: 11
    
    # Additional tests:
    print(sol.maxArrayValue([1, 2, 3]))  # Merges to [6], Output: 6
    print(sol.maxArrayValue([3, 2, 1]))  # No merge possible, Output: 3