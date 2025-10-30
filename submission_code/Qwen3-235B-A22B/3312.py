class Solution:
    def countKeyChanges(self, s: str) -> int:
        if len(s) < 2:
            return 0
        prev = s[0].lower()
        count = 0
        for i in range(1, len(s)):
            current = s[i].lower()
            if current != prev:
                count += 1
            prev = current
        return count