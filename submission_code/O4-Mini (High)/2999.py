class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # We can only swap indices (0,2) among evens and (1,3) among odds.
        # Thus s2 is reachable from s1 iff the multiset of characters
        # at even positions matches, and likewise for odd positions.
        return (sorted([s1[0], s1[2]]) == sorted([s2[0], s2[2]])) and \
               (sorted([s1[1], s1[3]]) == sorted([s2[1], s2[3]]))