class Solution:
    def countKeyChanges(self, s: str) -> int:
        if len(s) == 1:
            return 0
        total = 0
        last = s[0].lower()
        for i in range(1, len(s)):
            if s[i].lower() != last:
                last = s[i].lower()
                total += 1
        return total