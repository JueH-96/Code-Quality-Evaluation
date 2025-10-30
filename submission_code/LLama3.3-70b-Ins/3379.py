class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        This function calculates the score of a given string.
        The score is defined as the sum of the absolute difference between the ASCII values of adjacent characters.

        Args:
            s (str): The input string.

        Returns:
            int: The score of the string.
        """
        
        # Initialize the score to 0
        score = 0
        
        # Iterate over the characters in the string, excluding the last one
        for i in range(len(s) - 1):
            # Calculate the absolute difference between the ASCII values of the current and next characters
            diff = abs(ord(s[i]) - ord(s[i + 1]))
            
            # Add the difference to the score
            score += diff
        
        # Return the total score
        return score