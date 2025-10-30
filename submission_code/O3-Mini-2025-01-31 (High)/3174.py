class Solution:
    def minChanges(self, s: str) -> int:
        # We want to make s "beautiful". A beautiful string can be partitioned into
        # even-length segments that are homogeneous (all 0's or all 1's).
        # Notice that if s is beautiful, then one valid partition is to divide s into
        # consecutive pairs (since s has even length). In each pair, both characters must be identical.
        #
        # Thus, for any even-length index pair, if s[i] != s[i+1], we must flip one of them.
        # The minimal number of flips is simply the count of pairs (s[i], s[i+1]) that differ.
        #
        # For example:
        # s = "1001":
        #   Pair (0,1): "1" and "0" differ --> 1 flip required.
        #   Pair (2,3): "0" and "1" differ --> 1 flip required.
        # Total = 2, which is the answer.
        
        changes = 0
        n = len(s)
        # Process the string in steps of 2 because we want to form pairs.
        for i in range(0, n, 2):
            if s[i] != s[i+1]:
                changes += 1
        return changes