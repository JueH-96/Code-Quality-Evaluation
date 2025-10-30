from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x: int, y: int) -> int:
            prefix_len = 0
            while x > 0 and y > 0 and x % 10 == y % 10:
                x //= 10
                y //= 10
                prefix_len += 1
            return prefix_len

        max_prefix_length = 0

        for x in arr1:
            for y in arr2:
                max_prefix_length = max(max_prefix_length, common_prefix_length(x, y))

        return max_prefix_length