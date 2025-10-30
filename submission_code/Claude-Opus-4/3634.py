class Solution:
    def calculateScore(self, s: str) -> int:
        n = len(s)
        marked = [False] * n
        score = 0
        
        for i in range(n):
            if marked[i]:
                continue
                
            # Calculate mirror character
            mirror = chr(ord('z') + ord('a') - ord(s[i]))
            
            # Find closest unmarked index j < i with s[j] = mirror
            for j in range(i - 1, -1, -1):
                if not marked[j] and s[j] == mirror:
                    # Mark both indices
                    marked[i] = True
                    marked[j] = True
                    # Add to score
                    score += i - j
                    break
        
        return score