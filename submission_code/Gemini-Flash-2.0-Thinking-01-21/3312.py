class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        count = 0
        for i in range(1, len(s)):
            if s[i].lower() != s[i-1].lower():
                count += 1
        return count