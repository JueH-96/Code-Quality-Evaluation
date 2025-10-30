class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        return num.rstrip('0') if num != '0' else '0'