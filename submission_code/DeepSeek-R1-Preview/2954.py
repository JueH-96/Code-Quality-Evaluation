from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0

        freq = defaultdict(int)
        current_sum = 0

        # Initialize the first window
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] += 1

        max_sum = current_sum if len(freq) >= m else 0

        # Slide the window
        for i in range(k, n):
            # Remove the left element
            left_element = nums[i - k]
            current_sum -= left_element
            freq[left_element] -= 1
            if freq[left_element] == 0:
                del freq[left_element]

            # Add the current element
            current_element = nums[i]
            current_sum += current_element
            freq[current_element] += 1

            # Check if current window is almost unique and update max_sum
            if len(freq) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum

        return max_sum