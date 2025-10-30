from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers to strings for easier prefix manipulation.
        s1 = [str(x) for x in arr1]
        s2 = [str(x) for x in arr2]
        
        # The longest possible common prefix can only be as long as the shortest number 
        # (in terms of digits) from each array.
        max_len1 = max(len(x) for x in s1)
        max_len2 = max(len(x) for x in s2)
        L = min(max_len1, max_len2)
        
        # Check from the largest possible prefix length down to 1.
        for l in range(L, 0, -1):
            # Only consider numbers that have at least l digits.
            prefixes1 = {x[:l] for x in s1 if len(x) >= l}
            prefixes2 = {x[:l] for x in s2 if len(x) >= l}
            
            # If there's at least one prefix that exists in both sets, return l.
            if prefixes1 & prefixes2:
                return l
        
        # If no common prefix is found among any pairs, return 0.
        return 0