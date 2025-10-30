class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Convert the binary string to a list of digits
        digits = list(s)
        
        # Find the index of the rightmost '1'
        right_one_idx = len(digits) - 1 - digits[::-1].index('1')
        
        # Swap the rightmost '1' with the leftmost '0'
        for i in range(right_one_idx):
            if digits[i] == '0':
                digits[i], digits[right_one_idx] = digits[right_one_idx], digits[i]
                break
        
        # Return the resulting binary number as a string
        return ''.join(digits)