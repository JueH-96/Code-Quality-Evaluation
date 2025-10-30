class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        Calculates the score of a string.
        The score is defined as the sum of the absolute difference
        between the ASCII values of adjacent characters.

        Args:
            s: The input string consisting of lowercase English letters.
               Constraints: 2 <= s.length <= 100.

        Returns:
            The calculated score as an integer.
        """
        total_score = 0
        
        # Iterate through the string from the second character to the end.
        # For each character s[i], compare it with the previous character s[i-1].
        for i in range(1, len(s)):
            # Get the ASCII values of the current and previous characters.
            ascii_current = ord(s[i])
            ascii_previous = ord(s[i-1])
            
            # Calculate the absolute difference and add it to the total score.
            total_score += abs(ascii_current - ascii_previous)
            
        return total_score