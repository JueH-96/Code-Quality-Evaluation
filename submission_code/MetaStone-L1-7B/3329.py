from collections import defaultdict
from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_length = 8  # since the maximum possible digits in the numbers is 8
        for l in range(max_length, 0, -1):
            # Create a dictionary for arr1's prefixes of length l
            prefixes1 = defaultdict(int)
            for num in arr1:
                s = str(num)
                prefix = s[:l] if len(s) >= l else s
                prefixes1[prefix] += 1
            
            # Create a dictionary for arr2's prefixes of length l
            prefixes2 = defaultdict(int)
            for num in arr2:
                s = str(num)
                prefix = s[:l] if len(s) >= l else s
                prefixes2[prefix] += 1
            
            # Check if there's any common prefix between the two dictionaries
            for prefix in prefixes1:
                if prefix in prefixes2:
                    return l
        
        return 0  # if no common prefix found for any l >= 1