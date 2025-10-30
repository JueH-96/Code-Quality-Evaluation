class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        end_index = len(num) - 1
        while end_index >= 0 and num[end_index] == '0':
            end_index -= 1
        return num[:end_index + 1]