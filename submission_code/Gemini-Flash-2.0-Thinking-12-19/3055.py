class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = 0
        count_zeros = 0
        for char in s:
            if char == '1':
                count_ones += 1
            else:
                count_zeros += 1
        
        if count_ones == 0:
            return "" # Should not happen according to problem description

        result = ""
        if count_ones > 1:
            result += '1' * (count_ones - 1)
        result += '0' * count_zeros
        result += '1'
        return result