class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        count = 0
        prev_char = s[0].lower()
        
        for i in range(1, len(s)):
            curr_char = s[i].lower()
            if curr_char != prev_char:
                count += 1
                prev_char = curr_char
        
        return count