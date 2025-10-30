from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        """
        Counts the number of pairs (i, j) such that i < j and
        words[i] is both a prefix and a suffix of words[j].
        """
        n = len(words)
        count = 0

        # Iterate through all unique pairs of indices (i, j) where i < j.
        for i in range(n):
            for j in range(i + 1, n):
                # Let str1 be words[i] and str2 be words[j].
                # The problem defines a boolean function isPrefixAndSuffix(str1, str2).
                # This is true if str1 is both a prefix and a suffix of str2.
                # Python's built-in string methods make this check straightforward.
                if words[j].startswith(words[i]) and words[j].endswith(words[i]):
                    count += 1
        
        return count