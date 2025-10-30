class Solution:
    def countKeyChanges(self, s: str) -> int:
        changes = 0
        for i in range(1, len(s)):
            if s[i].islower() != s[i-1].islower():
                changes += 1
        return changes