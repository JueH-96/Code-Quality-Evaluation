class Solution:
    def clearDigits(self, s: str) -> str:
        while True:
            # Find the first digit's index
            first_digit_idx = -1
            for i, c in enumerate(s):
                if c.isdigit():
                    first_digit_idx = i
                    break
            if first_digit_idx == -1:
                break  # no more digits, exit loop

            # Find the closest non-digit to the left of first_digit_idx
            j = -1
            for k in range(first_digit_idx - 1, -1, -1):
                if not s[k].isdigit():
                    j = k
                    break

            # Construct the new string after removing j and first_digit_idx
            s = s[:j] + s[j+1:first_digit_idx] + s[first_digit_idx+1:]
        
        return s