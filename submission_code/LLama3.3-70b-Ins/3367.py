from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x: int) -> int:
            """Encrypts an integer by replacing every digit with the largest digit."""
            # Convert the integer to a string to easily access each digit
            str_x = str(x)
            # Find the maximum digit in the string
            max_digit = max(str_x)
            # Replace every digit with the maximum digit
            encrypted_str = max_digit * len(str_x)
            # Convert the encrypted string back to an integer
            return int(encrypted_str)

        # Encrypt each number in the list and sum the results
        return sum(encrypt(num) for num in nums)