class Solution:
    def countKeyChanges(self, s: str) -> int:
        changes = 0
        last_key = s[0].lower()
        
        for char in s[1:]:
            current_key = char.lower()
            if current_key != last_key:
                changes += 1
                last_key = current_key
                
        return changes