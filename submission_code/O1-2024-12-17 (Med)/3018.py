class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        We can pick exactly one subset of indices in str1 to increment each char
        in that subset by 1 (z -> a). Since this is a single operation that can
        include any number of positions, each character in str1 can effectively
        be thought of as having two possible final states: itself or its 'next'
        character cyclically. We only need to check if str2 can be matched as a
        subsequence against this two-state version of str1.
        """
        i, j = 0, 0
        n, m = len(str1), len(str2)

        while i < n and j < m:
            # The 'next' character of str1[i], wrapping z -> a
            next_char = chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a'))
            
            # If str2[j] matches either str1[i] or the incremented version,
            # we advance the pointer in str2.
            if str2[j] == str1[i] or str2[j] == next_char:
                j += 1
            i += 1
        
        return j == m