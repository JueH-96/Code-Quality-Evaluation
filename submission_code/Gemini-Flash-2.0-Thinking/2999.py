class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the sets of characters at indices 0 and 2 are the same
        if set([s1[0], s1[2]]) != set([s2[0], s2[2]]):
            return False

        # Check if the sets of characters at indices 1 and 3 are the same
        if set([s1[1], s1[3]]) != set([s2[1], s2[3]]):
            return False

        return True