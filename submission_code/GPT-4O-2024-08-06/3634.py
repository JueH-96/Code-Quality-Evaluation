class Solution:
    def calculateScore(self, s: str) -> int:
        # Function to find the mirror character
        def mirror_char(c):
            return chr(ord('z') - (ord(c) - ord('a')))
        
        # Initialize score and a list to keep track of marked indices
        score = 0
        marked = [False] * len(s)
        
        # Iterate over each character in the string
        for i in range(len(s)):
            if marked[i]:
                continue  # Skip if already marked
            
            # Find the mirror character of s[i]
            mirror = mirror_char(s[i])
            
            # Search for the closest unmarked index j < i with s[j] == mirror
            for j in range(i - 1, -1, -1):
                if not marked[j] and s[j] == mirror:
                    # Mark both indices i and j
                    marked[i] = True
                    marked[j] = True
                    # Add the difference to the score
                    score += i - j
                    break
        
        return score