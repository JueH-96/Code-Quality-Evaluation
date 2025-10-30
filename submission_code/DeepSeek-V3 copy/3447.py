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
                    # Remove the digit and the non-digit
                    del s[j]
                    del s[j]  # Since we removed one element, the next is now at j
                    i = max(j - 1, 0)
                else:
                    # If no non-digit to the left, just remove the digit
                    del s[i]
                    i = max(i - 1, 0)
            else:
                i += 1
        return ''.join(s)