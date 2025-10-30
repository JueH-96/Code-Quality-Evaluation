class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        prev_char = None
        for char in s:
            lower_char = char.lower()
            if prev_char is None:
                prev_char = lower_char
            elif lower_char != prev_char:
                count += 1
                prev_char = lower_char
        return count