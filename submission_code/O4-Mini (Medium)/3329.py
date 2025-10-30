from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers to strings to work with digit prefixes
        arr1_str = [str(x) for x in arr1]
        arr2_str = [str(x) for x in arr2]
        
        # Determine the maximum possible prefix length
        max_len1 = max(len(s) for s in arr1_str)
        max_len2 = max(len(s) for s in arr2_str)
        max_prefix_len = min(max_len1, max_len2)
        
        # Try prefix lengths from largest to smallest
        for L in range(max_prefix_len, 0, -1):
            # Collect all prefixes of length L from arr1
            prefixes1 = {s[:L] for s in arr1_str if len(s) >= L}
            if not prefixes1:
                continue
            
            # Check if any prefix of length L from arr2 matches
            for s in arr2_str:
                if len(s) >= L and s[:L] in prefixes1:
                    return L
        
        # No common prefix found
        return 0