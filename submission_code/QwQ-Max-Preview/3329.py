from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Create a set to store all prefixes of numbers in arr1
        prefixes = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefixes.add(s[:i])
        
        max_len = 0
        # Check each number in arr2 against the prefixes set
        for num in arr2:
            s = str(num)
            # Check from longest to shortest prefix
            for i in range(len(s), 0, -1):
                prefix = s[:i]
                if prefix in prefixes:
                    if i > max_len:
                        max_len = i
                    break  # Found the longest possible prefix for this number
        return max_len