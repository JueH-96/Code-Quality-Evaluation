class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove trailing zeros from the string
        return num.rstrip('0')