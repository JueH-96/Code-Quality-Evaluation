from typing import List # Required for the List[int] type hint

class Solution:
    def _encrypt(self, n: int) -> int:
        """
        Helper function to encrypt a single integer according to the rule.
        Example: encrypt(523) = 555
        """
        s_n = str(n)
        
        # Find the largest digit character in the string representation of n.
        # For example, max("523") returns '5'.
        largest_digit_char = max(s_n)
        
        # Determine the number of digits.
        num_digits = len(s_n)
        
        # Create the encrypted number string by repeating the largest digit.
        # For example, if largest_digit_char is '5' and num_digits is 3,
        # this forms "555" (Python's string multiplication: '5' * 3).
        encrypted_s = largest_digit_char * num_digits
        
        # Convert the encrypted string back to an integer.
        return int(encrypted_s)

    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        """
        Calculates the sum of encrypted elements from the nums list.
        """
        total_sum = 0
        for num in nums:
            # Encrypt each number and add it to the total sum.
            total_sum += self._encrypt(num)
            
        return total_sum