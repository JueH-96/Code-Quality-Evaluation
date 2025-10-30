class Solution:
    def countKeyChanges(self, s: str) -> int:
        if len(s) <= 1:
            return 0
        count = 0
        prev = s[0].lower()
        for c in s[1:]:
            current = c.lower()
            if current != prev:
                count += 1
            prev = current
        return count