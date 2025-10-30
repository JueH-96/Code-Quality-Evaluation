class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        if not num:
            return num
        
        i = len(num) - 1
        while i >= 0 and num[i] == '0':
            i -= 1
        
        return num[:i+1] if i+1 > 0 else "0"