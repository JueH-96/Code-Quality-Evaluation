class Solution:
    def countKeyChanges(self, s: str) -> int:
        s_lower = s.lower()
        changes = 0
        for i in range(1, len(s_lower)):
            if s_lower[i] != s_lower[i-1]:
                changes += 1
        return changes