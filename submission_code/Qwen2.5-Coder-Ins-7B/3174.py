class Solution:
    def minChanges(self, s: str) -> int:
        changes = 0
        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i] != s[i + 1]:
                changes += 1
            i += 2
        return changes