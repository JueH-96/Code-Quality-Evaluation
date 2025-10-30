from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        """
        Calculates the sum of encrypted integers from a given list.

        The encryption process for a number involves finding its largest digit
        and replacing all of its digits with this largest digit.
        """
        
        total_encrypted_sum = 0
        
        for num in nums:
            # Convert the number to a string to easily access its digits.
            s_num = str(num)
            
            # Find the largest digit. The max() function on a string of digits
            # correctly identifies the character of the largest digit.
            max_digit = max(s_num)
            
            # Create the encrypted number as a string by repeating the
            # largest digit for the length of the original number.
            encrypted_str = max_digit * len(s_num)
            
            # Convert the encrypted string back to an integer and add it to the total sum.
            total_encrypted_sum += int(encrypted_str)
            
        return total_encrypted_sum