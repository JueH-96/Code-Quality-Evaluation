from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # The numbers are at most 10**8  ->  9 digits
        MAX_DIGITS = 9
        
        # pref1[i]  (resp. pref2[i]) will contain every distinct prefix of length i
        # that appears in arr1  (resp. arr2)
        pref1 = [set() for _ in range(MAX_DIGITS + 1)]
        pref2 = [set() for _ in range(MAX_DIGITS + 1)]
        
        # fill prefixes coming from arr1
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                pref1[i].add(s[:i])
        
        # fill prefixes coming from arr2
        for num in arr2:
            s = str(num)
            for i in range(1, len(s) + 1):
                pref2[i].add(s[:i])
        
        # Check from longer prefix length to shorter
        for length in range(MAX_DIGITS, 0, -1):
            if pref1[length] and pref2[length]:
                # Intersection test – early-exit with the smaller set
                if len(pref1[length]) < len(pref2[length]):
                    small, large = pref1[length], pref2[length]
                else:
                    small, large = pref2[length], pref1[length]
                
                for p in small:
                    if p in large:
                        return length
        
        return 0