class Solution:
    def minChanges(self, s: str) -> int:
        # Since s has an even length, we can split it into pairs of 2.
        # A beautiful string must have each even-length substring with all same characters.
        # In particular, each pair itself is an even-length substring.
        # For each pair, if the two bits differ, a single change makes them equal.
        # Thus, the minimum changes is the count of pairs where the bits differ.
        changes = 0
        for i in range(0, len(s), 2):
            if s[i] != s[i+1]:
                changes += 1
        return changes