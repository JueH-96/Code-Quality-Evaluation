class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            zeros = 0
            ones = 0
            for j in range(i, n):
                # Count zeros and ones as we extend the substring
                if s[j] == '0':
                    zeros += 1
                else:  # s[j] == '1'
                    ones += 1
                
                # Check if the condition holds: ones >= zeros^2
                if ones >= zeros * zeros:
                    count += 1
        
        return count