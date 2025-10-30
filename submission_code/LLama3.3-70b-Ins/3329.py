from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x: int, y: int) -> int:
            """Return the length of the common prefix between two integers."""
            x_str = str(x)
            y_str = str(y)
            length = 0
            for a, b in zip(x_str, y_str):
                if a == b:
                    length += 1
                else:
                    break
            return length

        max_length = 0
        for x in arr1:
            for y in arr2:
                max_length = max(max_length, common_prefix_length(x, y))
        return max_length