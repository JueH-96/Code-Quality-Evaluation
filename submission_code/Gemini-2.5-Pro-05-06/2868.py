import collections
from typing import List # Usually provided by LeetCode's environment if List[int] is in stub

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        ans = 0
        left = 0
        
        # min_dq stores indices d_1, d_2, ..., d_k such that:
        # 1. d_1 < d_2 < ... < d_k (indices are increasing)
        # 2. nums[d_1] < nums[d_2] < ... < nums[d_k] (values are strictly increasing)
        # nums[min_dq[0]] is the minimum element in the current valid window.
        min_dq = collections.deque()
        
        # max_dq stores indices d_1, d_2, ..., d_k such that:
        # 1. d_1 < d_2 < ... < d_k (indices are increasing)
        # 2. nums[d_1] > nums[d_2] > ... > nums[d_k] (values are strictly decreasing)
        # nums[max_dq[0]] is the maximum element in the current valid window.
        max_dq = collections.deque()
        
        N = len(nums)

        for right in range(N):
            # Maintain min_dq:
            # Remove indices from the right of min_dq whose elements are >= nums[right].
            while min_dq and nums[min_dq[-1]] >= nums[right]:
                min_dq.pop()
            min_dq.append(right)
            
            # Maintain max_dq:
            # Remove indices from the right of max_dq whose elements are <= nums[right].
            while max_dq and nums[max_dq[-1]] <= nums[right]:
                max_dq.pop()
            max_dq.append(right)

            # Shrink window from the left if the condition (max_val - min_val <= 2) is violated.
            # nums[max_dq[0]] is the maximum in the current window elements tracked by deques.
            # nums[min_dq[0]] is the minimum in the current window elements tracked by deques.
            while nums[max_dq[0]] - nums[min_dq[0]] > 2:
                # The element nums[left] is about to be removed from the window [left..right].
                # If this element is currently the reported minimum or maximum, 
                # it must be popped from the respective deque.
                if min_dq[0] == left:
                    min_dq.popleft()
                if max_dq[0] == left:
                    max_dq.popleft()
                # Shrink the window by moving the left pointer.
                left += 1
            
            # After the while loop, the window nums[left..right] satisfies the condition.
            # All subarrays ending at 'right' and starting at an index 'k' 
            # where 'left' <= k <= 'right' are continuous.
            # These subarrays are: nums[left..right], nums[left+1..right], ..., nums[right..right].
            # There are (right - left + 1) such subarrays. Add this count to the total.
            ans += (right - left + 1)
            
        return ans