class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Use rstrip to remove all trailing '0' characters
        return num.rstrip('0')