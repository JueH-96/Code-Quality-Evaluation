import collections
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        left = 0
        
        # Deques to maintain the minimum and maximum elements in the current window [left, right]
        # min_dq stores indices of elements in increasing order of their values.
        # max_dq stores indices of elements in decreasing order of their values.
        min_dq = collections.deque()
        max_dq = collections.deque()
        
        for right in range(n):
            # 1. Maintain min_dq (monotonically increasing values)
            # Remove elements from the right of min_dq that are greater than or equal to nums[right]
            # because nums[right] is a smaller or equal value at a later index, making previous larger/equal values redundant for minimum tracking.
            while min_dq and nums[min_dq[-1]] >= nums[right]:
                min_dq.pop()
            min_dq.append(right)
            
            # 2. Maintain max_dq (monotonically decreasing values)
            # Remove elements from the right of max_dq that are less than or equal to nums[right]
            # because nums[right] is a larger or equal value at a later index, making previous smaller/equal values redundant for maximum tracking.
            while max_dq and nums[max_dq[-1]] <= nums[right]:
                max_dq.pop()
            max_dq.append(right)
            
            # 3. Check the window validity condition: max_val - min_val <= 2
            # If the condition is violated, shrink the window from the left.
            while nums[max_dq[0]] - nums[min_dq[0]] > 2:
                # Move the left pointer one step to the right.
                left += 1
                # If the element at the front of min_dq is now outside the window [left, right], remove it.
                if min_dq[0] < left:
                    min_dq.popleft()
                # If the element at the front of max_dq is now outside the window [left, right], remove it.
                if max_dq[0] < left:
                    max_dq.popleft()
            
            # 4. At this point, the window [left, right] is continuous.
            # Any subarray nums[k...right] where left <= k <= right is also continuous.
            # The number of such valid subarrays ending at 'right' is (right - left + 1).
            count += (right - left + 1)
            
        return count