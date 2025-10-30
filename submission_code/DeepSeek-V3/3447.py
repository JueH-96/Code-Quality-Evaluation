class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        i = 0
        while i < len(s):
            if s[i].isdigit():
                # Find the closest non-digit to the left
                j = i - 1
                while j >= 0 and s[j].isdigit():
                    j -= 1
                if j >= 0:
                    # Remove the digit and the closest non-digit to the left
                    del s[j]
                    del s[j]  # Since we removed one element, the index shifts
                    i = j  # Reset i to the new position after deletion
                else:
                    # If no non-digit to the left, just remove the digit
                    del s[i]
            else:
                i += 1
        return ''.join(s)