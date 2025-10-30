class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        n = len(num)
        i = n - 1
        while i >= 0 and num[i] == '0':
            i -= 1
        return num[:i+1]