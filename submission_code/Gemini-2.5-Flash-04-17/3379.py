class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        Calculates the score of a string, which is the sum of the absolute differences
        between the ASCII values of adjacent characters.

        Args:
            s: The input string.

        Returns:
            The calculated score.
        """
        # Initialize the total score
        total_score = 0
        
        # Iterate through the string considering adjacent pairs
        # The loop goes up to the second-to-last character (index len(s) - 2)
        # because for each index i, we access s[i] and s[i+1]
        for i in range(len(s) - 1):
            # Get the ASCII values of the current and next character
            ascii_current = ord(s[i])
            ascii_next = ord(s[i+1])
            
            # Calculate the absolute difference between the ASCII values
            # and add it to the total score
            total_score += abs(ascii_current - ascii_next)
            
        # Return the final calculated score
        return total_score