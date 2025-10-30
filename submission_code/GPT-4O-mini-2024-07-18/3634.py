class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        marked = [False] * len(s)
        
        # Function to get the mirror character
        def mirror(c):
            return chr(ord('a') + ord('z') - ord(c))
        
        for i in range(len(s)):
            if marked[i]:
                continue
            
            # Find the closest unmarked index j such that s[j] is the mirror of s[i]
            for j in range(i - 1, -1, -1):
                if not marked[j] and s[j] == mirror(s[i]):
                    marked[i] = True
                    marked[j] = True
                    score += i - j
                    break
        
        return score