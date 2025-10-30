class Solution:
    def scoreOfString(self, s: str) -> int:
        # Initialize score to 0
        score = 0
        
        # Iterate through the string and add the absolute difference
        # of the ASCII values of each pair of consecutive characters
        for i in range(len(s) - 1):
            score += abs(ord(s[i]) - ord(s[i + 1]))
            
        return score