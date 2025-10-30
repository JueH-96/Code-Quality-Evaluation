class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointer approach:
        # Since we can, in a single operation, increment any subset of characters of str1,
        # each character of str1 can effectively be either itself or its "next" character
        # in the cyclic alphabet (z -> a). We only need to check if str2 can be matched
        # under these conditions using a standard two-pointer subsequence check.
        
        def next_char(c: str) -> str:
            # Returns the next character cyclically (e.g. 'z' -> 'a')
            return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
        
        i, j = 0, 0  # i for str1, j for str2
        while i < len(str1) and j < len(str2):
            # If current str2 character matches directly or via one increment, advance j
            if str2[j] == str1[i] or str2[j] == next_char(str1[i]):
                j += 1
            i += 1
        
        return j == len(str2)