class Solution:
    def clearDigits(self, s: str) -> str:
        while any(char.isdigit() for char in s):
            for i, char in enumerate(s):
                if char.isdigit():
                    # Find the closest non-digit character to the left
                    left_index = i - 1
                    while left_index >= 0 and s[left_index].isdigit():
                        left_index -= 1
                    # Remove the digit and the closest non-digit character
                    s = s[:left_index] + s[i + 1:]
                    break
        return s