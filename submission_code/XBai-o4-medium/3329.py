from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        prefixes = set()
        # Generate all prefixes for arr1
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefixes.add(s[:i])
        
        max_len = 0
        # Check all prefixes for arr2 against the set
        for num in arr2:
            t = str(num)
            for j in range(1, len(t) + 1):
                prefix = t[:j]
                if prefix in prefixes:
                    if j > max_len:
                        max_len = j
                else:
                    # Since once a prefix is not found, longer prefixes including it won't be found
                    # So we can break early for this number
                    break
        return max_len