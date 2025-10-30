class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Convert the string to an integer to automatically remove trailing zeros
        integer_value = int(num)
        # Convert the integer back to a string and return
        return str(integer_value)