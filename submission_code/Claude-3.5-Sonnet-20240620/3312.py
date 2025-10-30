class Solution:
    def countKeyChanges(self, s: str) -> int:
        if len(s) <= 1:
            return 0
        
        changes = 0
        prev_char = s[0].lower()
        
        for char in s[1:]:
            if char.lower() != prev_char:
                changes += 1
                prev_char = char.lower()
        
        return changes