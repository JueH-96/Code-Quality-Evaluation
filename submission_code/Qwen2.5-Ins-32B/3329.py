from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def prefixes(num):
            s = str(num)
            return {s[:i] for i in range(1, len(s) + 1)}
        
        prefix_set1 = set()
        for num in arr1:
            prefix_set1.update(prefixes(num))
        
        max_len = 0
        for num in arr2:
            for prefix in prefixes(num):
                if prefix in prefix_set1:
                    max_len = max(max_len, len(prefix))
        
        return max_len