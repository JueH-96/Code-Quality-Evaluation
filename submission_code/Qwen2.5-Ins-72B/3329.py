from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(a: str, b: str) -> int:
            min_length = min(len(a), len(b))
            for i in range(min_length):
                if a[i] != b[i]:
                    return i
            return min_length
        
        max_length = 0
        arr1_str = [str(num) for num in arr1]
        arr2_str = [str(num) for num in arr2]
        
        for num1 in arr1_str:
            for num2 in arr2_str:
                current_length = common_prefix_length(num1, num2)
                max_length = max(max_length, current_length)
        
        return max_length