class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        # Try all possible starting positions
        for i in range(n):
            zeros = 0
            ones = 0
            
            # Extend substring from position i
            for j in range(i, n):
                # Add current character to counts
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if current substring has dominant ones
                if ones >= zeros * zeros:
                    count += 1
        
        return count