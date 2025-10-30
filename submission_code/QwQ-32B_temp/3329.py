from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Collect all prefixes from arr1
        prefix_set = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s)+1):
                prefix = s[:i]
                prefix_set.add(prefix)
        
        max_length = 0
        # Check each number in arr2 for the longest prefix in the set
        for num in arr2:
            s = str(num)
            # Check from longest to shortest prefix
            for i in range(len(s), 0, -1):
                prefix = s[:i]
                if prefix in prefix_set:
                    if i > max_length:
                        max_length = i
                    break  # No need to check shorter prefixes for this number
        return max_length