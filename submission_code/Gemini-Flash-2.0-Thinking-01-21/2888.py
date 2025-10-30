from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1

        counts = Counter(nums)
        dominant_element = -1
        for num, count in counts.items():
            if count * 2 > n:
                dominant_element = num
                break

        if dominant_element == -1:
            return -1

        for i in range(n - 1):
            left_subarray = nums[:i+1]
            right_subarray = nums[i+1:]

            left_count = 0
            for x in left_subarray:
                if x == dominant_element:
                    left_count += 1
            right_count = 0
            for x in right_subarray:
                if x == dominant_element:
                    right_count += 1

            if left_count * 2 > len(left_subarray) and right_count * 2 > len(right_subarray):
                return i

        return -1