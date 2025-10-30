class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert the numbers to strings and pad with zeros if necessary
        num1 = str(num1).zfill(4)
        num2 = str(num2).zfill(4)
        num3 = str(num3).zfill(4)

        # Initialize the key as an empty string
        key = ''

        # For each digit in the numbers
        for i in range(4):
            # Append the smallest digit among the i^th digits of num1, num2, and num3 to the key
            key += min(num1[i], num2[i], num3[i])

        # Return the key as an integer
        return int(key)