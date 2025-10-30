class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        Calculates the score of a string based on the sum of absolute differences
        of ASCII values between adjacent characters.
        
        Args:
            s: The input string, consisting of lowercase English letters.
               The length of s is between 2 and 100, inclusive.
               
        Returns:
            The score of the string.
        """
        
        score = 0
        # Iterate through the string to consider each pair of adjacent characters.
        # The loop runs from the first character to the second-to-last character,
        # so that s[i+1] is always a valid index.
        for i in range(len(s) - 1):
            # Add the absolute difference of ASCII values of the adjacent characters
            # s[i] and s[i+1] to the total score.
            score += abs(ord(s[i]) - ord(s[i+1]))
            
        return score