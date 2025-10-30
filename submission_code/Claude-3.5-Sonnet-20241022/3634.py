class Solution:
    def calculateScore(self, s: str) -> int:
        # Function to get mirror of a character
        def get_mirror(c):
            return chr(ord('a') + (25 - (ord(c) - ord('a'))))
        
        n = len(s)
        marked = [False] * n  # Track marked indices
        total_score = 0
        
        # Iterate through string from left to right
        for i in range(n):
            if marked[i]:  # Skip if current index is already marked
                continue
                
            # Find closest unmarked mirror character to the left
            found = False
            for j in range(i-1, -1, -1):
                if not marked[j] and s[j] == get_mirror(s[i]):
                    # Mark both indices
                    marked[i] = True
                    marked[j] = True
                    # Add to score
                    total_score += i - j
                    found = True
                    break
                    
            if not found:
                continue
                
        return total_score