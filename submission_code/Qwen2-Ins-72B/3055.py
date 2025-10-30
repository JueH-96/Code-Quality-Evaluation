class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        ones = s.count('1')
        # Move all ones to the front except the last one
        result = ['1'] * ones
        result[-1] = '0'  # Make the last one 0 to ensure the number is odd
        result += ['0'] * (len(s) - ones)
        return ''.join(result)