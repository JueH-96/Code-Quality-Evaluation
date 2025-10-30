from typing import List
from collections import defaultdict

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)

        # Find the dominant element
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1

        dominant_element = None
        for num, count in freq.items():
            if count * 2 > n:
                dominant_element = num
                break

        # If no dominant element is found, return -1
        if dominant_element is None:
            return -1

        # Initialize variables to track the count of the dominant element
        left_count = 0
        right_count = freq[dominant_element]

        # Iterate to find the minimum valid split index
        for i in range(n):
            if nums[i] == dominant_element:
                left_count += 1
                right_count -= 1

            # Check if the current split is valid
            if left_count * 2 > (i + 1) and right_count * 2 > (n - i - 1):
                return i

        return -1