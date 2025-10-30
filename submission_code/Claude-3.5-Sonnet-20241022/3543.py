class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # Check all possible substrings
        for i in range(n):
            zeros = 0
            ones = 0
            for j in range(i, n):
                # Count zeros and ones in current substring
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                    
                # Check if substring satisfies k-constraint
                if zeros <= k or ones <= k:
                    count += 1
                    
        return count