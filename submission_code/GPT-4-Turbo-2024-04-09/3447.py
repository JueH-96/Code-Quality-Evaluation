class Solution:
    def clearDigits(self, s: str) -> str:
        while any(char.isdigit() for char in s):
            for i, char in enumerate(s):
                if char.isdigit():
                    # Find the closest non-digit character to the left
                    j = i - 1
                    while j >= 0 and s[j].isdigit():
                        j -= 1
                    if j >= 0:
                        # Remove the digit and the closest non-digit character to its left
                        s = s[:j] + s[i+1:]
                    else:
                        # If no non-digit character to the left, just remove the digit
                        s = s[i+1:]
                    break
        return s