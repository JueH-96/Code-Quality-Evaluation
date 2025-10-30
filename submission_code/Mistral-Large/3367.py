from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x: int) -> int:
            # Convert the number to a string to iterate over each digit
            digits = str(x)
            # Find the largest digit in the number
            max_digit = max(digits)
            # Replace every digit with the largest digit
            encrypted = int(max_digit * len(digits))
            return encrypted

        # Sum up all encrypted elements
        return sum(encrypt(num) for num in nums)