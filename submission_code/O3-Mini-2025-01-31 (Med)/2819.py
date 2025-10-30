class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Remove all trailing zeros using rstrip
        # Note: For a string "0", rstrip will yield an empty string, so we handle that case.
        result = num.rstrip('0')
        return result if result != '' else '0'