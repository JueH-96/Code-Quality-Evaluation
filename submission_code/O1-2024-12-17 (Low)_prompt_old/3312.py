class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        changes = 0
        
        # Compare each character with the previous one, ignoring case
        for i in range(1, len(s)):
            if s[i].lower() != s[i - 1].lower():
                changes += 1
        
        return changes