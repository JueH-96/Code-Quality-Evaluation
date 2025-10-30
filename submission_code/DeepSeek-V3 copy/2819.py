class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Iterate from the end of the string to find the first non-zero character
        i = len(num) - 1
        while i >= 0 and num[i] == '0':
            i -= 1
        # Return the substring from the start to the first non-zero character
        return num[:i+1]