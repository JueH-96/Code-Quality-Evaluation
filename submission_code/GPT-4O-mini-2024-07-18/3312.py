class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        changes = 0
        last_char = s[0].lower()  # Start with the first character in lowercase
        
        for char in s[1:]:
            current_char = char.lower()
            if current_char != last_char:  # Check if the current character (lowercase) is different from the last
                changes += 1
                last_char = current_char  # Update the last character to the current one
        
        return changes