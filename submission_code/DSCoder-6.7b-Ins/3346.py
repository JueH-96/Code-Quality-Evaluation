class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        # Initialize the result as 'a's
        res = ['a'] * n
        k -= n  # Subtract n from k

        # Start from the end of the string
        i = n - 1

        # While k > 0
        while k > 0:
            # Calculate the remaining value of k
            rem = min(25, k)

            # Add the remaining value to the current character
            res[i] = chr(ord(res[i]) + rem)

            # Subtract the remaining value from k
            k -= rem

            # Move to the previous character
            i -= 1

        # Return the result as a string
        return ''.join(res)