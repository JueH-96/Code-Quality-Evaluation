class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Strip all '0' characters from the right end of the string
        return num.rstrip('0')