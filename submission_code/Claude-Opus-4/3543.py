class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # For each starting position
        for i in range(n):
            zeros = 0
            ones = 0
            
            # Extend the window from position i
            for j in range(i, n):
                # Update counts
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if current substring satisfies k-constraint
                if zeros <= k or ones <= k:
                    count += 1
                else:
                    # No point checking longer substrings starting at i
                    break
        
        return count