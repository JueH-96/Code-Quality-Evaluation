class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        n = len(s)
        
        # Find all positions of character c in the string
        positions = [i for i in range(n) if s[i] == c]
        
        # Calculate the number of substrings starting and ending with c
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                count += 1
        
        return count