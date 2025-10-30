from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0 # left pointer of the sliding window [i, j]

        # Deques to store indices for tracking minimum and maximum in the current window [i, j]
        # min_deque stores indices k in increasing order of nums[k]
        min_deque = deque()
        # max_deque stores indices k in decreasing order of nums[k]
        max_deque = deque()

        for j in range(n): # right pointer of the sliding window [i, j]
            # Maintain the monotonic increasing deque (for minimum)
            # Remove indices from the end whose values are >= nums[j]
            while min_deque and nums[min_deque[-1]] >= nums[j]:
                min_deque.pop()
            # Add the current index j
            min_deque.append(j)

            # Maintain the monotonic decreasing deque (for maximum)
            # Remove indices from the end whose values are <= nums[j]
            while max_deque and nums[max_deque[-1]] <= nums[j]:
                max_deque.pop()
            # Add the current index j
            max_deque.append(j)

            # Check the condition: max(window [i, j]) - min(window [i, j]) <= 2
            # The minimum in the current window [i, j] is nums[min_deque[0]]
            # The maximum in the current window [i, j] is nums[max_deque[0]]
            # The indices at the head of the deques (min_deque[0], max_deque[0])
            # always point to the min/max elements currently within the window [i, j].
            # This is guaranteed because when 'i' increments, we remove indices == old_i.
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # The current window [i, j] is invalid. We need to shrink it from the left.
                # Increment the left pointer i.
                # Before incrementing i, check if the index i is currently the head of
                # either deque. If so, it means the element at index i was the min/max
                # of the window [i, j], and it's being removed from the window [i+1, j].
                if min_deque[0] == i:
                    min_deque.popleft()
                if max_deque[0] == i:
                    max_deque.popleft()
                # Now increment the left pointer
                i += 1

            # After the while loop, the window [i, j] is valid.
            # It is the largest continuous subarray ending at j that starts at or after the current i.
            # All subarrays ending at j and starting at k, where i <= k <= j, are also valid.
            # The number of such subarrays is the length of the current valid window: j - i + 1.
            count += (j - i + 1)

        return count