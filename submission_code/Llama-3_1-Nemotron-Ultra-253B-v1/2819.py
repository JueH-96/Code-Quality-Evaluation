class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        last_non_zero = len(num) - 1
        while last_non_zero >= 0 and num[last_non_zero] == '0':
            last_non_zero -= 1
        return num[:last_non_zero + 1]