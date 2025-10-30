class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while any(c.isdigit() for c in s_list):
            # Find the first occurrence of a digit
            d = next(i for i, char in enumerate(s_list) if char.isdigit())
            # Find the closest non-digit to the left of the digit
            l = d - 1
            while l >= 0 and s_list[l].isdigit():
                l -= 1
            # Remove the characters from l to d inclusive
            del s_list[l:d+1]
        return ''.join(s_list)