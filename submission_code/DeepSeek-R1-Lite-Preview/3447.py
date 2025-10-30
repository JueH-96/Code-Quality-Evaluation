class Solution:
    def clearDigits(self, s: str) -> str:
        chars = list(s)
        while any(c.isdigit() for c in chars):
            # Find the first digit
            first_digit_index = next(i for i, c in enumerate(chars) if c.isdigit())
            # Find the closest non-digit to its left
            left_index = first_digit_index - 1
            while left_index >= 0 and chars[left_index].isdigit():
                left_index -= 1
            # Delete the digit and the non-digit to its left
            del chars[first_digit_index]
            del chars[left_index]
        return ''.join(chars)