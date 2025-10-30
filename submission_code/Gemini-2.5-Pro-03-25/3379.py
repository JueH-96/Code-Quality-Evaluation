import math

class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        Calculates the score of a string based on the sum of absolute differences
        between ASCII values of adjacent characters.

        Args:
            s: The input string.

        Returns:
            The score of the string.
        """
        total_score = 0
        # Iterate through the string from the first character up to the second-to-last
        # The loop runs s.length - 1 times.
        for i in range(len(s) - 1):
            # Get the ASCII values of the current character and the next character
            ascii_val1 = ord(s[i])
            ascii_val2 = ord(s[i+1])
            
            # Calculate the absolute difference between the ASCII values
            diff = abs(ascii_val1 - ascii_val2)
            
            # Add the difference to the total score
            total_score += diff
            
        return total_score

# Example usage (for testing outside the class structure if needed):
# sol = Solution()
# print(sol.scoreOfString("hello")) # Output: 13
# print(sol.scoreOfString("zaz"))   # Output: 50