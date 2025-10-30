import math

class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        """
        Counts the number of symmetric integers in the range [low, high].
        An integer x is symmetric if it has an even number of digits (2n)
        and the sum of the first n digits equals the sum of the last n digits.
        Numbers with an odd number of digits are never symmetric.
        
        Constraints:
        1 <= low <= high <= 10^4
        """
        count = 0  # Initialize the counter for symmetric integers
        
        # Iterate through each number in the range [low, high] inclusive
        for num in range(low, high + 1):
            s_num = str(num)  # Convert the number to a string to access digits
            n = len(s_num)    # Get the number of digits
            
            # A number can only be symmetric if it has an even number of digits
            # Check if the number of digits is even
            if n % 2 == 0:
                mid = n // 2  # Calculate the midpoint index (half the length)
                
                # Calculate the sum of the first half of the digits.
                # s_num[:mid] gets the first half of the string.
                # map(int, ...) converts each character digit to an integer.
                # sum(...) calculates the sum of these integers.
                sum_first_half = sum(map(int, s_num[:mid]))
                
                # Calculate the sum of the second half of the digits.
                # s_num[mid:] gets the second half of the string.
                sum_second_half = sum(map(int, s_num[mid:]))
                
                # Check if the sum of the first half equals the sum of the second half
                if sum_first_half == sum_second_half:
                    count += 1  # Increment the counter if the number is symmetric
                    
        return count  # Return the total count of symmetric integers found