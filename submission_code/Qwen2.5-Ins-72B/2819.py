class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Strip trailing zeros from the string
        return num.rstrip('0')