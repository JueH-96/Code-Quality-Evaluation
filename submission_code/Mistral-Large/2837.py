class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Check if it's possible to make num1 equal to 0
        if num1 + num2 < 0:
            return -1

        # Calculate the sum of num1 and num2
        total = num1 + num2

        # Count the number of 1s in the binary representation of total
        count_operations = bin(total).count('1')

        return count_operations