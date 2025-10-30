class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Two pointers: i over str1, j over str2
        i, j = 0, 0
        n, m = len(str1), len(str2)
        # Helper to get the next character cyclically
        def nxt(c: str) -> str:
            # ord('a') = 97, ord('z') = 122
            o = ord(c) - 97
            o = (o + 1) % 26
            return chr(o + 97)
        # Scan through str1 to try to match all of str2
        while i < n and j < m:
            c1 = str1[i]
            c2 = str2[j]
            # If c2 matches either the original or incremented char of str1[i],
            # we can match str2[j] here (by choosing to select or not select i in the operation).
            if c2 == c1 or c2 == nxt(c1):
                j += 1
            i += 1
        # If we've matched all of str2, it's possible
        return j == m