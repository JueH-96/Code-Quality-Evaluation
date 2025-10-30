class Solution:
    def countKeyChanges(self, s: str) -> int:
        if len(s) <= 1:
            return 0
        
        count = 0
        for i in range(1, len(s)):
            # Compare current and previous character ignoring case
            if s[i].lower() != s[i-1].lower():
                count += 1
        
        return count