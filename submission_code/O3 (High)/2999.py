class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        """
        We can only swap the characters inside the two independent pairs of indices
        (0, 2) and (1, 3).  Therefore, in any string reachable from s1:
            - the multiset of characters at indices 0 and 2 is fixed,
            - the multiset of characters at indices 1 and 3 is fixed.

        Hence s1 can be transformed into s2 (or vice-versa) iff the two strings
        contain exactly the same multiset in each of those pairs.
        """
        # compare multisets of the even-indexed pair (0, 2)
        if sorted((s1[0], s1[2])) != sorted((s2[0], s2[2])):
            return False
        # compare multisets of the odd-indexed pair (1, 3)
        if sorted((s1[1], s1[3])) != sorted((s2[1], s2[3])):
            return False
        return True