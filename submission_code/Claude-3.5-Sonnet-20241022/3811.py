class Solution:
    def reverseDegree(self, s: str) -> int:
        # Initialize result
        result = 0
        
        # For each character in the string
        for i, char in enumerate(s, 1):  # i is 1-indexed
            # Calculate reversed alphabet position (a=26, b=25, ..., z=1)
            rev_alpha_pos = 26 - (ord(char) - ord('a'))
            
            # Multiply character's reversed alphabet position with its string position
            result += rev_alpha_pos * i
            
        return result