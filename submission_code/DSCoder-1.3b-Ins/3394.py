class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Initialize the result as the maximum possible value of n
        result = n
        # Iterate from n-1 to 1
        for i in range(n-1, 0, -1):
            # Calculate the bitwise AND of the current number and the next number
            bitwise_and = i ^ (i + 1)
            # If the bitwise AND is equal to x, update the result
            if bitwise_and == x:
                result = i
                break
        # Return the result
        return result