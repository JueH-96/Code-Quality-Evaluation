class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        count = 0
        prev_char = s[0].lower()
        for char in s[1:]:
            current_char = char.lower()
            if current_char != prev_char:
                count += 1
            prev_char = current_char
        return count