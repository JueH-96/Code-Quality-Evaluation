class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # If str2 is longer than str1, it can't be a subsequence.
        if len(str2) > len(str1):
            return False

        j = 0  # pointer into str2
        n2 = len(str2)

        # Scan through str1, trying to match str2[j]
        for c in str1:
            if j >= n2:
                break
            # Compute difference ord(str2[j]) - ord(c)
            # A match is possible if:
            #   diff == 0   -> c == str2[j]
            #   diff == 1   -> c incremented by 1 matches str2[j]
            #   diff == -25 -> wrap from 'z' to 'a'
            d = ord(str2[j]) - ord(c)
            if d == 0 or d == 1 or d == -25:
                j += 1

        return j == n2