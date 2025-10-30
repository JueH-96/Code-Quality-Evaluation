class Solution:
    def countKeyChanges(self, s: str) -> int:
        s_lower = s.lower()
        count = 0
        prev = s_lower[0]
        for c in s_lower[1:]:
            if c != prev:
                count += 1
            prev = c
        return count