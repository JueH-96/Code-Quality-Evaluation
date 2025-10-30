class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Iterate from the end of the string
        i = len(num) - 1
        while i >= 0 and num[i] == '0':
            i -= 1
        
        # Return the substring from the beginning to the last non-zero digit
        return num[:i+1]