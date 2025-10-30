class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        changes = 0
        last_char = s[0].lower()  # Convert first character to lowercase for uniformity
        
        for i in range(1, len(s)):
            current_char = s[i].lower()  # Convert current character to lowercase
            if current_char != last_char:
                changes += 1
                last_char = current_char
        
        return changes