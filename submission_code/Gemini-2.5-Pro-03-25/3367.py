import math
from typing import List # Ensure this import is present for the List type hint

class Solution:
  """
  Implements the solution to find the sum of encrypted integers as per the problem specification.
  Encryption rule: Replace every digit in a number 'x' with the largest digit present in 'x'.
  """
  def sumOfEncryptedInt(self, nums: List[int]) -> int:
    """
    Calculates the sum of encrypted elements for the given list of positive integers.

    Args:
        nums: A list of positive integers. Example: [1, 2, 3] or [10, 21, 31].

    Returns:
        An integer representing the sum of the encrypted values of all numbers in the input list.
        Example for [1, 2, 3]: encrypt(1)=1, encrypt(2)=2, encrypt(3)=3. Sum = 1+2+3 = 6.
        Example for [10, 21, 31]: encrypt(10)=11, encrypt(21)=22, encrypt(31)=33. Sum = 11+22+33 = 66.
    """
    
    total_sum = 0 # Initialize the accumulator for the sum of encrypted numbers

    # Iterate through each number in the input list 'nums'
    for num in nums:
        # Convert the integer to its string representation to easily work with digits.
        s_num = str(num)
        
        # Find the largest digit present in the number.
        # The max() function on a string of digits returns the character representing the largest digit.
        # For example, max("523") returns '5'.
        max_digit_char = '0' # Initialize with '0', the smallest possible digit character
        for digit_char in s_num:
            if digit_char > max_digit_char:
                max_digit_char = digit_char
        # Alternatively, using the built-in max function:
        # max_digit_char = max(s_num) 

        # Determine the number of digits in the original number.
        # This is equivalent to the length of the string representation.
        num_digits = len(s_num)
        
        # Construct the string representation of the encrypted number.
        # This involves repeating the largest digit character 'num_digits' times.
        # Example: If max_digit_char is '5' and num_digits is 3, encrypted_str becomes "555".
        encrypted_str = max_digit_char * num_digits
        
        # Convert the encrypted number string back to an integer.
        # Example: int("555") results in the integer 555.
        encrypted_num = int(encrypted_str)
        
        # Add the obtained encrypted number to the total sum.
        total_sum += encrypted_num
            
    # After processing all numbers in the list, return the final computed sum.
    return total_sum