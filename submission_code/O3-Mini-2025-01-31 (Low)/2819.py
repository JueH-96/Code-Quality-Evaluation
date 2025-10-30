class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove trailing zeros by using rstrip method for strings
        return num.rstrip('0')