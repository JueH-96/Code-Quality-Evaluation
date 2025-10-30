class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        if len(s) == 0:
            return 0
        
        prev = s[0].lower()
        for i in range(1, len(s)):
            curr = s[i].lower()
            if curr != prev:
                count += 1
                prev = curr
        return count