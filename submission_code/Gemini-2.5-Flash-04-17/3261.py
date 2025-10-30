from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # This function checks if it's possible to partition nums into at least n-k segments
        # such that the bitwise AND of elements within each segment is a submask of target_mask.
        # If it returns True, it means we can achieve a final bitwise OR value Y
        # such that Y is a submask of target_mask.
        def check(target_mask: int) -> bool:
            segments = 0
            # Represents the accumulated bitwise AND of the current potential segment
            # Initialize with a mask with all relevant bits set (0 to 29), as nums[i] < 2^30.
            current_and = (1 << 30) - 1

            for x in nums:
                current_and &= x
                # If the current accumulated AND value is a submask of target_mask,
                # we can end a segment here. To maximize the number of segments,
                # we greedily end the segment as soon as the condition is met.
                # (current_and | target_mask) == target_mask is the check for submask.
                if (current_and | target_mask) == target_mask:
                    segments += 1
                    current_and = (1 << 30) - 1 # Reset for the next segment

            # The greedy strategy maximizes the number of segments whose AND is a submask of target_mask.
            # If the maximum number of such segments is >= n-k, then it is possible.
            return segments >= n - k

        # Binary search for the minimum possible OR value.
        # The possible values are between 0 and 2^30 - 1.
        low = 0
        high = (1 << 30) - 1 # Max possible OR value < 2^30
        ans = high # Initialize ans with max possible value

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                # It is possible to achieve a final OR value Y such that Y is a submask of mid.
                # The minimum possible OR value is <= mid.
                ans = mid
                high = mid - 1
            else:
                # It is NOT possible to achieve a final OR value Y such that Y is a submask of mid.
                # This means any achievable OR value must have at least one bit set
                # that is 0 in mid. Therefore, the minimum possible OR value must be > mid.
                low = mid + 1

        return ans