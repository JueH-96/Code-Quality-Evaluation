class Solution:
    def isFascinating(self, n: int) -> bool:
        # Convert the numbers to strings for easier concatenation and checking
        num_str = str(n)
        double_str = str(2 * n)
        triple_str = str(3 * n)
        
        # Concatenate the numbers
        concatenated_str = num_str + double_str + triple_str
        
        # Check if the concatenated string has a length of 9 (since it should contain digits 1-9 exactly once)
        if len(concatenated_str) != 9:
            return False
        
        # Check if the concatenated string contains all digits from 1 to 9 exactly once
        for digit in '123456789':
            if concatenated_str.count(digit) != 1:
                return False
        
        # If all checks pass, the number is fascinating
        return True