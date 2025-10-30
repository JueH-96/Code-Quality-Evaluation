from typing import List
from collections import deque

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        def sum_of_min_max(nums, k):
            n = len(nums)
            total_sum = 0

            for window_size in range(1, k + 1):
                min_deque = deque()
                max_deque = deque()

                for i in range(n):
                    # Remove elements not within the current window
                    if min_deque and min_deque[0] <= i - window_size:
                        min_deque.popleft()
                    if max_deque and max_deque[0] <= i - window_size:
                        max_deque.popleft()

                    # Maintain the min deque
                    while min_deque and nums[min_deque[-1]] >= nums[i]:
                        min_deque.pop()
                    min_deque.append(i)

                    # Maintain the max deque
                    while max_deque and nums[max_deque[-1]] <= nums[i]:
                        max_deque.pop()
                    max_deque.append(i)

                    # Calculate the sum of min and max for the current window
                    if i >= window_size - 1:
                        total_sum += nums[min_deque[0]] + nums[max_deque[0]]

            return total_sum

        return sum_of_min_max(nums, k)