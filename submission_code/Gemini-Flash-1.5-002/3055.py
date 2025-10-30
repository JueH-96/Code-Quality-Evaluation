class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        if count_ones == 0:
            return "0" #Should not happen based on constraints but handle for robustness.

        count_zeros = len(s) - count_ones
        result = ""
        
        #Handle the case where there are no zeros.
        if count_zeros == 0:
            return s[:-1] + s[-1]

        result += '1' * (count_ones -1)
        result += '0' * count_zeros
        result += '1'
        return result