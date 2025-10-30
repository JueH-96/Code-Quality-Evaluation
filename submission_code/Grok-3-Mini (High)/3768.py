class Solution:
    def hasSameDigits(self, s: str) -> bool:
        digits = [int(char) for char in s]
        while len(digits) > 2:
            new_digits = []
            for i in range(len(digits) - 1):
                sum_val = (digits[i] + digits[i + 1]) % 10
                new_digits.append(sum_val)
            digits = new_digits
        return digits[0] == digits[1]