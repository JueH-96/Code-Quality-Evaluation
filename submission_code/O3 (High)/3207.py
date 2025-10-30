class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        """
        The only operation allowed is deleting the last character of a string (if its
        length is at least 2).  After any sequence of such deletions every string is
        forced to become a prefix of its initial value.  Therefore, for the three
        strings to end up equal they must share a non-empty common prefix that we
        can keep while deleting everything to its right.

        Let L be the length of the longest common prefix of the three strings.
        • If L == 0 there is no non-empty common prefix ⇒ impossible ⇒ return -1.
        • Otherwise we keep that prefix and delete the remaining characters.
          The number of deletions needed is:
              (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
          =  len(s1) + len(s2) + len(s3) - 3·L
        This value is minimal because keeping the longest possible prefix
        minimises the total amount removed.
        """
        # Length of the longest common prefix of the three strings
        lcp = 0
        for a, b, c in zip(s1, s2, s3):
            if a == b == c:
                lcp += 1
            else:
                break

        if lcp == 0:                  # no non-empty common prefix
            return -1

        # minimal number of deletions
        return len(s1) + len(s2) + len(s3) - 3 * lcp