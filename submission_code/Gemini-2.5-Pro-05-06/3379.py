class Solution:
    def scoreOfString(self, s: str) -> int:
        """
        Calculates the score of a string s.
        The score is defined as the sum of the absolute difference between 
        the ASCII values of adjacent characters.
        
        Constraints:
        - 2 <= s.length <= 100
        - s consists only of lowercase English letters.
        """
        
        total_score = 0
        
        # Iterate through the string from the first character up to the
        # second-to-last character. This allows us to consider pairs
        # of adjacent characters (s[i], s[i+1]).
        # The loop will run for (len(s) - 1) iterations.
        # For example, if s = "hello" (length 5), len(s)-1 = 4.
        # The loop range(4) means i will take values 0, 1, 2, 3.
        # When i = 0, we consider s[0] and s[1].
        # When i = 1, we consider s[1] and s[2].
        # ...
        # When i = len(s)-2, we consider s[len(s)-2] and s[len(s)-1].
        
        # The constraint 2 <= s.length guarantees that len(s)-1 is at least 1,
        # so the loop range(len(s)-1) will not be empty and will execute at least once.
        
        for i in range(len(s) - 1):
            # ord(character) returns the ASCII value of the character.
            ascii_val1 = ord(s[i])
            ascii_val2 = ord(s[i+1])
            
            # abs() calculates the absolute difference.
            difference = abs(ascii_val1 - ascii_val2)
            
            # Add this difference to the total score.
            total_score += difference
            
        return total_score