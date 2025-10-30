from typing import List

class Solution:
    def _isPrefixAndSuffix(self, str1: str, str2: str) -> bool:
        """
        Checks if str1 is both a prefix and a suffix of str2.
        """
        # Python's startswith() and endswith() methods correctly handle cases
        # where str1 might be longer than str2 (they return False in such cases).
        
        # Check if str1 is a prefix of str2
        is_prefix = str2.startswith(str1)
        
        # Check if str1 is a suffix of str2
        is_suffix = str2.endswith(str1)
        
        # str1 must be both a prefix and a suffix
        return is_prefix and is_suffix

    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        n = len(words)
        count = 0
        
        # Iterate over all unique pairs (i, j) such that i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Let words[i] be str1 and words[j] be str2
                str1 = words[i]
                str2 = words[j]
                
                # If words[i] is both a prefix and a suffix of words[j]
                if self._isPrefixAndSuffix(str1, str2):
                    count += 1
                    
        return count