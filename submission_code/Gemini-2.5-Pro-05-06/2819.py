class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Use the rstrip() method to remove all trailing '0' characters.
        # Given the constraints (num is a positive integer string, no leading zeros),
        # num will always contain at least one non-zero digit, so rstrip('0')
        # will not produce an empty string.
        return num.rstrip('0')