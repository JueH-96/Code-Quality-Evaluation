class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        num1 = 0  # Sum of integers not divisible by m
        num2 = 0  # Sum of integers divisible by m

        # Iterate through all integers from 1 to n (inclusive)
        for i in range(1, n + 1):
            # Check if the current integer i is divisible by m
            if i % m == 0:
                # If divisible, add to num2
                num2 += i
            else:
                # If not divisible, add to num1
                num1 += i
        
        # Return the difference between num1 and num2
        return num1 - num2