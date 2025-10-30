class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        for i in range(1, len(s)):
            if s[i-1].lower() != s[i].lower():
                count += 1
        return count