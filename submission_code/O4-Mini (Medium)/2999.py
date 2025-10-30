class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # We can only swap positions 0<->2 and 1<->3 in each string.
        # Thus, the characters at even indices (0,2) can only permute among themselves,
        # and the characters at odd indices (1,3) can only permute among themselves.
        # Check that the multisets of even-indexed chars and odd-indexed chars match.
        return (sorted([s1[0], s1[2]]) == sorted([s2[0], s2[2]])
                and sorted([s1[1], s1[3]]) == sorted([s2[1], s2[3]]))