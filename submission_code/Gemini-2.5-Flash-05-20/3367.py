from typing import List

class Solution:
    def encrypt(self, x: int) -> int:
        """
        Replaces every digit in x with the largest digit in x.
        For example, encrypt(523) = 555 and encrypt(213) = 333.
        """
        s_x = str(x)
        
        # Find the largest digit character in the string representation of x.
        # For example, for "523", max('5', '2', '3') is '5'.
        # For "213", max('2', '1', '3') is '3'.
        max_digit_char = '0'
        for digit_char in s_x:
            if digit_char > max_digit_char:
                max_digit_char = digit_char
        # A more concise way: max_digit_char = max(s_x)
        
        # Determine the number of digits in x.
        num_digits = len(s_x)
        
        # Construct the encrypted number.
        # This involves repeating the max_digit_char num_digits times
        # and then converting the resulting string to an integer.
        # For example, if max_digit_char = '5' and num_digits = 3,
        # max_digit_char * num_digits results in the string "555".
        # int("555") gives 555.
        encrypted_s = max_digit_char * num_digits
        return int(encrypted_s)

    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        """
        Returns the sum of encrypted elements in the given list nums.
        """
        total_sum = 0
        for num in nums:
            total_sum += self.encrypt(num) # Call the helper method to encrypt each number
        return total_sum