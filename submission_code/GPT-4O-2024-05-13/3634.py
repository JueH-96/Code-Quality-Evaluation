class Solution:
    def calculateScore(self, s: str) -> int:
        # Create a dictionary to map each letter to its mirror
        mirror = {chr(i): chr(219 - i) for i in range(97, 123)}
        
        # Initialize the score and a list to keep track of marked indices
        score = 0
        marked = [False] * len(s)
        
        # Iterate through the string
        for i in range(len(s)):
            if marked[i]:
                continue
            for j in range(i - 1, -1, -1):
                if not marked[j] and s[j] == mirror[s[i]]:
                    marked[i] = True
                    marked[j] = True
                    score += i - j
                    break
        
        return score