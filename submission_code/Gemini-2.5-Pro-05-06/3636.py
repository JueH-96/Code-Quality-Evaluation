class Solution:
    def isBalanced(self, num: str) -> bool:
        sum_even_indices = 0
        sum_odd_indices = 0

        # Iterate through the string, getting both index and character
        # enumerate provides (index, value) pairs, e.g., (0, '1'), (1, '2'), ...
        for index, digit_char in enumerate(num):
            # Convert the character representation of a digit to an integer
            digit_val = int(digit_char)
            
            # Check if the current index is even or odd
            if index % 2 == 0:  # Indices 0, 2, 4, ... are even
                sum_even_indices += digit_val
            else:  # Indices 1, 3, 5, ... are odd
                sum_odd_indices += digit_val
        
        # A string is balanced if the sum of digits at even indices
        # is equal to the sum of digits at odd indices.
        return sum_even_indices == sum_odd_indices