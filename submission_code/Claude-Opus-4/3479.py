class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        # For each starting position
        for i in range(n):
            zeros = 0
            ones = 0
            
            # Expand to the right
            for j in range(i, n):
                # Update counts based on current character
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if substring s[i:j+1] has dominant ones
                # ones >= zeros * zeros
                if ones >= zeros * zeros:
                    count += 1
                    
                # Early termination: if zeros² > remaining length + ones
                # then we can't possibly satisfy the condition
                if zeros * zeros > n - i:
                    break
        
        return count