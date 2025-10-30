class Solution:
    def countKeyChanges(self, s: str) -> int:
        # If the string is empty or has only one character, there are no key changes.
        if len(s) < 2:
            return 0
        
        count = 0
        # Compare each character with the previous one in a case-insensitive way.
        for i in range(1, len(s)):
            if s[i].lower() != s[i-1].lower():
                count += 1
                
        return count