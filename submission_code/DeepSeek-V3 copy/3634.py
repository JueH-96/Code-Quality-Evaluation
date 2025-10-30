class Solution:
    def calculateScore(self, s: str) -> int:
        # Create a mapping from each letter to its mirror
        mirror = {}
        for i in range(26):
            mirror[chr(ord('a') + i)] = chr(ord('z') - i)
        
        # Initialize a list to keep track of marked indices
        marked = [False] * len(s)
        score = 0
        
        # Iterate through the string
        for i in range(len(s)):
            if not marked[i]:
                # Find the mirror character
                mirror_char = mirror[s[i]]
                # Iterate backwards to find the closest unmarked j
                for j in range(i-1, -1, -1):
                    if not marked[j] and s[j] == mirror_char:
                        # Mark both i and j
                        marked[i] = True
                        marked[j] = True
                        # Add to score
                        score += (i - j)
                        break
        
        return score