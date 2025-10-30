class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        prev_char = s[0].lower()
        for char in s[1:]:
            if char.lower() != prev_char:
                count += 1
                prev_char = char.lower()
        return count