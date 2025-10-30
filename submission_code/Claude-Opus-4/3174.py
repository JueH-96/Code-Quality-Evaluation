class Solution:
    def minChanges(self, s: str) -> int:
        changes = 0
        # Process the string in pairs
        for i in range(0, len(s), 2):
            # If the pair has different characters, we need one change
            if s[i] != s[i + 1]:
                changes += 1
        return changes