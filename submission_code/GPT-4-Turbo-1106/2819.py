class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Use rstrip method to remove trailing zeros
        return num.rstrip('0')