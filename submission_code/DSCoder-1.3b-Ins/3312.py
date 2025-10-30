class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        prev_char = ''
        for char in s:
            if char != prev_char:
                count += 1
            prev_char = char
        return count