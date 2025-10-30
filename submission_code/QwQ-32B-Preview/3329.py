from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Step 1: Generate all prefixes from arr1 and store in a set
        prefixes_arr1 = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefix = s[:i]
                prefixes_arr1.add(prefix)
        
        # Step 2: Initialize maximum length
        max_length = 0
        
        # Step 3: Iterate through arr2 and find common prefixes
        for num in arr2:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefix = s[:i]
                if prefix in prefixes_arr1:
                    if i > max_length:
                        max_length = i
                    # Since we're iterating from shortest to longest,
                    # we can break early once the longest prefix is found
                    # if i == len(s):
                    #     break
        
        return max_length