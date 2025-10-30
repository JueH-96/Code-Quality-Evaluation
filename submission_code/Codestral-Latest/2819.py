class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove trailing zeros by stripping '0' characters from the end of the string
        return num.rstrip('0')