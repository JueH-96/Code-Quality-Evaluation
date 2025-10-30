class Solution:
    def isBalanced(self, num: str) -> bool:
        if not num or len(num) < 2 or not num.isdigit():
            raise ValueError("Invalid input: num must be a string of digits with length >=2")

        even_sum = 0
        odd_sum = 0
        for i, digit in enumerate(num):
            digit = int(digit)
            if (i + 1) % 2 == 0:  # Even index
                even_sum += digit
            else:  # Odd index
                odd_sum += digit
        return even_sum == odd_sum