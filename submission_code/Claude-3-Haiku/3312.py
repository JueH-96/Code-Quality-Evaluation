class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        prev_char = None
        
        for char in s:
            if prev_char is None or (prev_char.isupper() != char.isupper()):
                count += 1
            prev_char = char
        
        return count