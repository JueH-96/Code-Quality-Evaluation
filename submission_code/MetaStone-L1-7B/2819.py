class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        i = len(num) - 1
        while i >= 0:
            if num[i] != '0':
                break
            i -= 1
        if i == -1:
            return '0'
        else:
            return num[:i+1]