class Solution:
    def minChanges(self, s: str) -> int:
        # Since every beautiful substring must have even length and be all 0's or all 1's,
        # it never reduces the total number of changes to merge two adjacent even-length
        # monochrome blocks into a larger block. Thus it's optimal to split s into
        # contiguous pairs and make each pair uniform.
        #
        # For each pair (s[i], s[i+1]), the cost is 0 if they are already equal,
        # otherwise 1 (we flip one character to match the other).
        n = len(s)
        changes = 0
        # walk in steps of 2
        for i in range(0, n, 2):
            if s[i] != s[i+1]:
                changes += 1
        return changes