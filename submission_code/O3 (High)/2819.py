class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove all '0' characters from the right side of the string.
        # str.rstrip removes characters from the end until a different character is met.
        return num.rstrip('0')