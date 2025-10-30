class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # Iterate over all possible starting points of substrings
        for start in range(n):
            zeros = 0
            ones = 0
            
            # Iterate over all possible ending points of substrings starting from 'start'
            for end in range(start, n):
                if s[end] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if the current substring satisfies the k-constraint
                if zeros <= k or ones <= k:
                    count += 1
                else:
                    # If the current substring does not satisfy the k-constraint,
                    # no need to check further substrings starting from 'start'
                    break
        
        return count