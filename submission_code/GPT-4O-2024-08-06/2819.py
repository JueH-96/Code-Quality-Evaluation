class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Strip the trailing zeros from the string
        return num.rstrip('0')