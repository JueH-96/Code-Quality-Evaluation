class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove trailing zeros
        while num[-1] == '0':
            num = num[:-1]
        return num