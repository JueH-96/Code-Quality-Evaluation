class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        We want to check if str2 can be a subsequence of str1 after at most one operation.
        The operation allows choosing any subset of indices in str1 and incrementing those
        characters cyclically ('z' -> 'a'). Since we can pick any subset (including empty),
        effectively, for each position i in str1, we can either leave it as is or increment
        it by 1.

        This boils down to a two-pointer check, where for each character in str1, we see
        if it can match the current character of str2 in either its original form or its
        incremented form. If yes, we move forward in str2; if not, we just move on in str1.
        If we can advance through all characters of str2, the answer is True; otherwise False.
        """

        # Two pointers: i for str1, j for str2
        i, j = 0, 0
        n, m = len(str1), len(str2)

        while i < n and j < m:
            # Original character
            ch1 = str1[i]
            # Incremented (cyclic) character
            inc_ch1 = chr((ord(ch1) - ord('a') + 1) % 26 + ord('a'))

            # Check if current str1 character or its incremented version matches str2[j]
            if ch1 == str2[j] or inc_ch1 == str2[j]:
                j += 1  # Move to the next character in str2
            i += 1      # Move to the next character in str1

        return j == m