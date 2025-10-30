from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Build prefix sets for arr1 and arr2, keyed by prefix length
        prefixes1 = {}
        prefixes2 = {}
        
        # Helper to populate prefix dict
        def add_prefixes(num: int, dct: dict):
            s = str(num)
            # For each possible prefix length
            for k in range(1, len(s) + 1):
                pref = s[:k]
                if k not in dct:
                    dct[k] = set()
                dct[k].add(pref)
        
        # Populate for both arrays
        for x in arr1:
            add_prefixes(x, prefixes1)
        for y in arr2:
            add_prefixes(y, prefixes2)
        
        # Check lengths from largest possible down to 1
        # The maximum prefix length is at most 10 (since numbers <= 1e8)
        max_len = 0
        # Determine the overall maximum length present in either dictionary
        if prefixes1:
            max_len = max(prefixes1.keys())
        if prefixes2:
            max_len = max(max_len, max(prefixes2.keys()))
        
        for length in range(max_len, 0, -1):
            if length in prefixes1 and length in prefixes2:
                # If there's any common prefix of this length
                if prefixes1[length] & prefixes2[length]:
                    return length
        
        return 0