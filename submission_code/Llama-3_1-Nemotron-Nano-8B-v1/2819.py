class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        end = len(num) - 1
        while end >= 0 and num[end] == '0':
            end -= 1
        return num[:end+1]