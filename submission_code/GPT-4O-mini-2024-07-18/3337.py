class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        positions = [i for i in range(len(s)) if s[i] == c]
        n = len(positions)
        
        # Calculate the number of substrings starting and ending with c
        for i in range(n):
            for j in range(i, n):
                count += 1
        
        return count