from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        # Helper function to encrypt a single integer according to the problem definition.
        def encrypt(x: int) -> int:
            """
            Replaces every digit in x with the largest digit in x.
            For example, encrypt(523) = 555 and encrypt(213) = 333.
            """
            # Convert the number to a string to easily access its digits.
            s = str(x)
            
            # Find the largest digit character within the string.
            # The max() function on a string compares characters based on their Unicode values.
            # For digits '0' through '9', this correctly identifies the largest digit.
            max_digit_char = max(s)
            
            # Determine the number of digits in the original number.
            num_digits = len(s)
            
            # Construct the encrypted number as a string.
            # This involves repeating the largest digit character 'num_digits' times.
            # For example, if max_digit_char is '5' and num_digits is 3, the string becomes "555".
            encrypted_s = max_digit_char * num_digits
            
            # Convert the resulting encrypted string back into an integer and return it.
            return int(encrypted_s)
        
        # Initialize a variable to hold the sum of the encrypted numbers.
        total_sum = 0
        
        # Iterate through each integer in the input list 'nums'.
        for num in nums:
            # Encrypt the current number using our helper function.
            encrypted_num = encrypt(num)
            
            # Add the encrypted number to the running total sum.
            total_sum += encrypted_num
            
        # After processing all numbers in the list, return the final total sum.
        return total_sum