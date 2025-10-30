class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We can only remove characters from the right.
        # Therefore, any mismatch in the leftmost characters of s1, s2, s3
        # makes it impossible to ever make them all equal (since we cannot
        # remove from the front or reduce a string to length 0).
        #
        # The largest possible final string that they can all share after
        # right-truncation is the longest common prefix (LCP) of s1, s2, s3.
        # Let that common prefix have length L. Then we remove all characters
        # in each string beyond the L-th character from the left.
        #
        # The cost (number of removal operations) is:
        #   (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
        #
        # If the three strings have no common starting character (L = 0),
        # then the answer is -1 because we cannot fix the leftmost mismatch
        # by removing characters from the right.

        n1, n2, n3 = len(s1), len(s2), len(s3)
        # Find the length of the longest common prefix
        m = min(n1, n2, n3)
        L = 0
        while L < m and s1[L] == s2[L] == s3[L]:
            L += 1

        # If no common prefix exists (i.e., L == 0),
        # it's impossible to make the strings equal
        if L == 0:
            return -1

        # Otherwise, the cost is the total number of removals to keep only
        # the first L characters in each string
        return (n1 - L) + (n2 - L) + (n3 - L)