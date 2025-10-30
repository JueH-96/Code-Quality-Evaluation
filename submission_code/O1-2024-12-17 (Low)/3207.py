class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We can only delete the rightmost character from any string whose length is >= 2.
        # Hence, effectively, to make the strings equal, they must share the same prefix of length L >= 1
        # (since we can never reduce any string to length 0).
        #
        # The plan:
        # 1) Let Lmax = min(len(s1), len(s2), len(s3)).
        # 2) Check possible lengths from Lmax down to 1.
        # 3) If s1[:L] == s2[:L] == s3[:L], then that prefix can be the final string.
        # 4) The cost (number of operations) = sum( len(si) - L ) for i in (1,2,3).
        # 5) If none of the lengths from Lmax down to 1 works, return -1.

        Lmax = min(len(s1), len(s2), len(s3))
        for L in range(Lmax, 0, -1):
            prefix1 = s1[:L]
            prefix2 = s2[:L]
            prefix3 = s3[:L]
            if prefix1 == prefix2 == prefix3:
                # We found the longest possible common prefix (length>=1)
                return (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
        # If we cannot find any common prefix of length>=1, return -1
        return -1