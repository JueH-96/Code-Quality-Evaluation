from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0

        max_sum = 0
        current_sum = 0
        distinct_count = 0
        element_count = defaultdict(int)

        # Helper function to add an element to the current window
        def add_element(num):
            nonlocal current_sum, distinct_count
            current_sum += num
            if element_count[num] == 0:
                distinct_count += 1
            element_count[num] += 1

        # Helper function to remove an element from the current window
        def remove_element(num):
            nonlocal current_sum, distinct_count
            current_sum -= num
            if element_count[num] == 1:
                distinct_count -= 1
            element_count[num] -= 1

        # Initialize the first window
        for i in range(k):
            add_element(nums[i])

        if distinct_count >= m:
            max_sum = current_sum

        # Slide the window
        for i in range(k, n):
            add_element(nums[i])
            remove_element(nums[i - k])

            if distinct_count >= m:
                max_sum = max(max_sum, current_sum)

        return max_sum